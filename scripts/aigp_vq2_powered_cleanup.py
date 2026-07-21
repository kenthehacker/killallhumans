"""Fail-closed cleanup-only fallback for the VQ2 powered calibration pilot.

The module is intentionally inert on import. Live process inspection, lease
takeover, and MAVLink adapter construction remain narrow injected boundaries
for offline tests. The production entry point constructs only retained
bootstrap handles before capability admission and defers lease and MAVLink
construction until that one-shot gate succeeds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from scripts import aigp_vq2_powered_attempt as attempt_contract
from scripts import aigp_vq2_powered_runtime as powered_runtime


ROLE = "cleanup_fallback"
CLEANUP_EPOCH = "fallback-cleanup-0"
CAPABILITY_DOMAIN = "aigp-vq2-powered-cleanup/1"
CAPABILITY_RELEASE_DURATION_NS = 3_000_000_000
OWNED_HANDLE_CLOSE_DURATION_NS = 2_000_000_000
STDERR_LIMIT_BYTES = 1_048_576
EXTERNAL_ANNOUNCE_INTERVAL_NS = 100_000_000
ZERO_COMMAND = {
    "roll_rate_rad_s": 0.0,
    "pitch_rate_rad_s": 0.0,
    "yaw_rate_rad_s": 0.0,
    "thrust": 0.0,
}

_RECEIVED_HEARTBEAT = "aigp-vq2-received-heartbeat/1"
_RECEIVED_RACE = "aigp-vq2-received-race-status/1"
_RECEIVED_IMU = "aigp-vq2-received-imu/1"


class CleanupFallbackError(RuntimeError):
    """Base class for fail-closed fallback failures."""


class CleanupBootstrapError(CleanupFallbackError):
    """The immutable attempt/process/capability gate was not proved."""


class CleanupExecutionError(CleanupFallbackError):
    """The admitted cleanup state machine could not prove cleanup."""


class CleanupDeadlineError(CleanupExecutionError):
    """An absolute fallback phase deadline was reached."""


class CleanupEvidenceError(CleanupExecutionError):
    """Cleanup evidence was incomplete or internally inconsistent."""


@dataclass(frozen=True)
class CleanupArguments:
    powered_attempt_envelope: str
    wrapper_process: str
    powered_process_authority: str
    cleanup_capability_handle: str
    parent_liveness_handle: str
    cleanup_certificate: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.aigp_vq2_powered_cleanup",
        allow_abbrev=False,
    )
    parser.add_argument("--powered-attempt-envelope", required=True)
    parser.add_argument("--wrapper-process", required=True)
    parser.add_argument("--powered-process-authority", required=True)
    parser.add_argument("--cleanup-capability-handle", required=True)
    parser.add_argument("--parent-liveness-handle", required=True)
    parser.add_argument("--cleanup-certificate", required=True)
    return parser


def parse_cleanup_arguments(argv: Sequence[str]) -> CleanupArguments:
    if type(argv) not in {list, tuple} or any(type(item) is not str for item in argv):
        raise TypeError("cleanup argv must be an exact string list or tuple")
    namespace = build_argument_parser().parse_args(list(argv))
    return CleanupArguments(
        powered_attempt_envelope=namespace.powered_attempt_envelope,
        wrapper_process=namespace.wrapper_process,
        powered_process_authority=namespace.powered_process_authority,
        cleanup_capability_handle=namespace.cleanup_capability_handle,
        parent_liveness_handle=namespace.parent_liveness_handle,
        cleanup_certificate=namespace.cleanup_certificate,
    )


class ProcessBoundary(Protocol):
    """Shared-runtime seam for retained Windows process/handle proof."""

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


@dataclass(frozen=True)
class LeaseProof:
    owner_role: str
    generation: int
    record_sha256: str
    authority_valid: bool
    takeover_completed_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        if type(self.owner_role) is not str or self.owner_role not in {
            "wrapper",
            "cleanup-fallback-parent-death",
        }:
            raise ValueError("cleanup lease owner role is invalid")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("cleanup lease generation must be nonnegative")
        if (
            type(self.record_sha256) is not str
            or len(self.record_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.record_sha256)
        ):
            raise ValueError("cleanup lease record hash is invalid")
        if type(self.authority_valid) is not bool:
            raise TypeError("cleanup lease authority state must be boolean")
        if self.takeover_completed_monotonic_ns is not None and (
            type(self.takeover_completed_monotonic_ns) is not int
            or self.takeover_completed_monotonic_ns < 0
        ):
            raise ValueError("cleanup takeover completion time is invalid")


class LeaseBoundary(Protocol):
    """Powered-ledger seam; implementations retain same-thread ownership."""

    def prove_live_delegation(
        self,
        *,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
    ) -> LeaseProof: ...

    def take_over_abandoned(
        self,
        *,
        role_secret: memoryview,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
        deadline_monotonic_ns: int,
    ) -> LeaseProof: ...

    def heartbeat_takeover(
        self,
        proof: LeaseProof,
        *,
        phase: str,
        deadline_monotonic_ns: int,
    ) -> LeaseProof: ...

    def release_takeover(
        self,
        proof: LeaseProof,
        *,
        deadline_monotonic_ns: int,
    ) -> bool: ...


@dataclass(frozen=True)
class ReceivedDatagram:
    payload: bytes
    source: tuple[str, int]

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("received datagram payload must be immutable bytes")
        if (
            type(self.source) is not tuple
            or len(self.source) != 2
            or type(self.source[0]) is not str
            or type(self.source[1]) is not int
            or not 1 <= self.source[1] <= 65_535
        ):
            raise ValueError("received datagram source must be one exact host/port pair")


@dataclass(frozen=True)
class CleanupDatagramDispatch:
    source_accepted: bool
    peer_frozen_now: bool
    rejected_source: bool
    malformed: bool
    production_dispatched: bool
    source_promoted: bool
    peer: tuple[str, int] | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_accepted",
            "peer_frozen_now",
            "rejected_source",
            "malformed",
            "production_dispatched",
            "source_promoted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        terminal = sum(
            (self.source_accepted, self.rejected_source, self.malformed)
        )
        if terminal not in {0, 1} or (
            terminal == 0 and self.failure_reason is None
        ):
            raise ValueError("cleanup datagram dispatch outcome is invalid")
        if self.production_dispatched and not self.source_accepted:
            raise ValueError("unaccepted cleanup datagram reached production")
        if self.peer is not None and (
            type(self.peer) is not tuple
            or len(self.peer) != 2
            or type(self.peer[0]) is not str
            or type(self.peer[1]) is not int
        ):
            raise ValueError("cleanup datagram peer is invalid")
        if self.failure_reason is not None and (
            type(self.failure_reason) is not str or not self.failure_reason
        ):
            raise ValueError("cleanup datagram failure reason is invalid")


@dataclass(frozen=True)
class NonattitudeDispatch:
    request_monotonic_ns: int
    outcome: str
    receipt: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if type(self.request_monotonic_ns) is not int or self.request_monotonic_ns < 0:
            raise ValueError("dispatch request time must be nonnegative")
        if self.outcome not in {"returned", "raised", "uncertain"}:
            raise ValueError("dispatch outcome is invalid")


@dataclass(frozen=True)
class ResetDispatch:
    request: NonattitudeDispatch
    boundary: Mapping[str, Any]


@dataclass(frozen=True)
class WorkerCloseProof:
    receiver_joined: bool
    announcer_joined: bool
    owned_handles_closed: bool

    def __post_init__(self) -> None:
        if not all(
            type(value) is bool
            for value in (
                self.receiver_joined,
                self.announcer_joined,
                self.owned_handles_closed,
            )
        ):
            raise TypeError("worker close proof fields must be exact booleans")


@dataclass(frozen=True)
class CleanupBackendAuthority:
    """Live callbacks and the one cleanup guard shared with the adapter."""

    outbound_guards: powered_runtime.PoweredOutboundGuards
    authorize_outbound: Callable[..., int]
    role_valid: Callable[[], bool]
    parent_alive: Callable[[], bool]
    lease_valid: Callable[[], bool]

    def __post_init__(self) -> None:
        if type(self.outbound_guards) is not powered_runtime.PoweredOutboundGuards:
            raise TypeError(
                "cleanup backend guard must be exact PoweredOutboundGuards"
            )
        for name in (
            "authorize_outbound",
            "role_valid",
            "parent_alive",
            "lease_valid",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"cleanup backend {name} must be callable")


class CleanupMavlinkBackend(Protocol):
    """Post-admission adapter seam with no camera or nonzero command method.

    The backend owns the one adapter/transport source authority and the sole
    receive call. Scratch-parser messages never cross this seam. The backend
    has no autonomous send authority: synchronous 100 ms announcements call
    the dispatcher supplied through :class:`CleanupBackendAuthority`.
    """

    @property
    def source_authority(self) -> powered_runtime.MavlinkSourceFreeze: ...

    def open(self, *, deadline_monotonic_ns: int) -> None: ...

    def receive_and_dispatch_datagram(
        self,
        max_wait_ns: int,
    ) -> CleanupDatagramDispatch | None: ...

    def drain_received_observations(self) -> Sequence[Mapping[str, Any]]: ...

    def send_exact_zero(
        self,
        command: Mapping[str, float],
        *,
        deadline_monotonic_ns: int,
    ) -> Mapping[str, Any]: ...

    def send_disarm(self, *, deadline_monotonic_ns: int) -> NonattitudeDispatch: ...

    def send_reset(
        self,
        *,
        baseline: Mapping[str, Any],
        deadline_monotonic_ns: int,
    ) -> ResetDispatch: ...

    def outbound_receipts(self) -> Sequence[Mapping[str, Any]]: ...

    def outbound_audit(self) -> Mapping[str, Any]: ...

    def collision_observations(self) -> Sequence[Mapping[str, Any]]: ...

    def request_stop(self) -> None: ...

    def join_workers(
        self,
        *,
        deadline_monotonic_ns: int,
        progress_callback: Callable[[], None] | None = None,
    ) -> WorkerCloseProof: ...


class CertificatePublisher(Protocol):
    def publish_create_new(
        self,
        path: str,
        value: Mapping[str, Any],
        *,
        deadline_monotonic_ns: int,
        progress_callback: Callable[[], None] | None = None,
    ) -> str: ...


@dataclass
class CleanupServices:
    process_boundary: ProcessBoundary
    lease_boundary: LeaseBoundary | None
    backend_factory: Callable[
        [
            powered_runtime.ExclusiveUdpEndpoint,
            CleanupBackendAuthority,
        ],
        CleanupMavlinkBackend,
    ]
    publisher: CertificatePublisher
    capability_operations: powered_runtime.CapabilityPipeOperations
    monotonic_ns: Callable[[], int]
    contract: Any = attempt_contract
    load_record: Callable[[str, Any], Mapping[str, Any]] | None = None
    endpoint_factory: Callable[..., powered_runtime.ExclusiveUdpEndpoint] = (
        powered_runtime.create_exclusive_udp_endpoint
    )
    lease_boundary_factory: Callable[["CleanupAdmission"], LeaseBoundary] | None = None
    owned_process_boundary: Any | None = field(default=None, repr=False)


class AIGPMavlinkCleanupBackend:
    """Cleanup-only bridge over the adapter's external powered receive mode."""

    def __init__(
        self,
        adapter: Any,
        *,
        zero_evidence_factory: Callable[..., Mapping[str, Any]],
        monotonic_ns: Callable[[], int],
    ) -> None:
        from competition.aigp_mavlink import (
            AIGPMavlinkAdapter,
            POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        )

        if type(adapter) is not AIGPMavlinkAdapter:
            raise TypeError("cleanup backend adapter must be exact AIGPMavlinkAdapter")
        if adapter.powered_receive_mode != POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP:
            raise ValueError("cleanup backend requires external cleanup receive mode")
        if not callable(zero_evidence_factory):
            raise TypeError("zero_evidence_factory must be callable")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        source = adapter.powered_source_authority
        if type(source) is not powered_runtime.MavlinkSourceFreeze:
            raise TypeError("cleanup adapter source authority is unavailable")
        self.adapter = adapter
        self._source_authority = source
        self.zero_evidence_factory = zero_evidence_factory
        self.monotonic_ns = monotonic_ns
        self._receipts: list[dict[str, Any]] = []
        self._collision_generation: int | None = None
        self._next_collision_sequence = 0
        self._collision_observations: list[dict[str, Any]] = []
        self._next_announcement_monotonic_ns: int | None = None
        self._opened = False
        self._stop_requested = False

    @property
    def source_authority(self) -> powered_runtime.MavlinkSourceFreeze:
        return self._source_authority

    def _now(self) -> int:
        return powered_runtime.read_qpc_ns(self.monotonic_ns)

    @staticmethod
    def _run(awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise CleanupExecutionError(
            "cleanup adapter call cannot run inside an active event loop"
        )

    @staticmethod
    def _primitive(value: Any) -> dict[str, Any]:
        to_primitive = getattr(value, "to_primitive", None)
        if not callable(to_primitive):
            raise CleanupEvidenceError("adapter evidence lacks to_primitive()")
        primitive = to_primitive()
        if type(primitive) is not dict:
            raise CleanupEvidenceError("adapter evidence primitive is not an object")
        return dict(primitive)

    def _capture_receipts(self) -> list[dict[str, Any]]:
        captured = [
            self._primitive(item)
            for item in self.adapter.drain_outbound_receipts()
        ]
        self._receipts.extend(captured)
        return captured

    def _begin_collision_generation(self, generation: Any) -> None:
        if type(generation) is not int or generation < 0:
            raise CleanupEvidenceError("collision generation is invalid")
        if self._collision_generation is None:
            self._collision_generation = generation
            self._next_collision_sequence = 0
            return
        if generation == self._collision_generation:
            return
        if generation != self._collision_generation + 1:
            raise CleanupEvidenceError("collision generation is discontinuous")
        self._collision_generation = generation
        self._next_collision_sequence = 0

    @staticmethod
    def _copy_collision_observation(value: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(value)
        copied["collision"] = dict(value["collision"])
        return copied

    def _record_collision(
        self,
        value: Any,
        *,
        generation: int,
        observed_monotonic_ns: int,
        phase: str,
        disposition: str,
    ) -> dict[str, Any]:
        if type(value) is not dict:
            raise CleanupEvidenceError("adapter collision payload is not an object")
        self._begin_collision_generation(generation)
        row = {
            "schema": "aigp-vq2-runner-collision-observation/1",
            "reset_generation": generation,
            "observation_sequence": self._next_collision_sequence,
            "host_clock_id": attempt_contract.HOST_CLOCK_ID,
            "observed_monotonic_ns": observed_monotonic_ns,
            "phase": phase,
            "disposition": disposition,
            "boundary": "runner_drain_not_receiver_receipt",
            "collision": dict(value),
        }
        try:
            checked = attempt_contract.validate_collision_observation(row)
        except BaseException as exc:
            raise CleanupEvidenceError(
                "adapter collision payload validation failed"
            ) from exc
        self._next_collision_sequence += 1
        retained = self._copy_collision_observation(checked)
        self._collision_observations.append(retained)
        return self._copy_collision_observation(retained)

    def open(self, *, deadline_monotonic_ns: int) -> None:
        if self._opened:
            raise CleanupExecutionError("cleanup adapter backend is single-use")
        self._run(
            self.adapter.connect(
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
        )
        if self.adapter.powered_receive_owner != "external_cleanup":
            raise CleanupEvidenceError(
                "cleanup adapter did not retain external receive ownership"
            )
        if self.adapter._rx_thread is not None or self.adapter._announce_thread is not None:
            raise CleanupEvidenceError("cleanup adapter created a competing worker")
        self._opened = True

    def _announce_if_due(self, now: int) -> None:
        if (
            self._stop_requested
            or self._source_authority.peer is None
            or self._source_authority.source_rejected_latched
            or self.adapter.powered_outbound_guards.cleanup_state == "closed"
        ):
            return
        if self._next_announcement_monotonic_ns is None:
            self._next_announcement_monotonic_ns = now
        if now < self._next_announcement_monotonic_ns:
            return
        self.adapter.announce_powered_external_cleanup()
        self._capture_receipts()
        periods = (
            (now - self._next_announcement_monotonic_ns)
            // EXTERNAL_ANNOUNCE_INTERVAL_NS
        ) + 1
        self._next_announcement_monotonic_ns += (
            periods * EXTERNAL_ANNOUNCE_INTERVAL_NS
        )

    def receive_and_dispatch_datagram(
        self,
        max_wait_ns: int,
    ) -> CleanupDatagramDispatch | None:
        if not self._opened or self._stop_requested:
            raise CleanupExecutionError("cleanup adapter receive is unavailable")
        if (
            type(max_wait_ns) is not int
            or not 1 <= max_wait_ns <= powered_runtime.MAX_POLL_INTERVAL_NS
        ):
            raise ValueError("cleanup receive wait must be within the poll bound")
        now = self._now()
        self._announce_if_due(now)
        wait_ns = max_wait_ns
        if self._next_announcement_monotonic_ns is not None:
            until_announcement = self._next_announcement_monotonic_ns - now
            if until_announcement > 0:
                wait_ns = min(wait_ns, until_announcement)
        value = self.adapter.receive_powered_external(max(1, wait_ns))
        after = self._now()
        if value is None:
            self._announce_if_due(after)
            return None
        if value.peer_frozen_now:
            self._announce_if_due(after)
        return CleanupDatagramDispatch(
            source_accepted=value.source_accepted,
            peer_frozen_now=value.peer_frozen_now,
            rejected_source=value.rejected_source,
            malformed=value.malformed,
            production_dispatched=value.production_dispatched,
            source_promoted=value.source_promoted,
            peer=value.peer,
            failure_reason=value.failure_reason,
        )

    def drain_received_observations(self) -> Sequence[Mapping[str, Any]]:
        return [
            self._primitive(item)
            for item in self.adapter.drain_received_observations()
        ]

    def _call_and_receipt(
        self,
        awaitable: Any,
        *,
        category: str,
        audit_name: str,
    ) -> tuple[int, str, dict[str, Any] | None, int, int]:
        audit_before = getattr(self.adapter.outbound_audit(), audit_name)
        request = self._now()
        raised = False
        try:
            self._run(awaitable)
        except BaseException:
            raised = True
        captured = self._capture_receipts()
        audit_after = getattr(self.adapter.outbound_audit(), audit_name)
        if category == "attitude_target":
            matching = [
                item
                for item in captured
                if item.get("schema")
                == "aigp-vq2-attitude-target-outbound/1"
            ]
        else:
            matching = [
                item for item in captured if item.get("category") == category
            ]
        receipt = matching[-1] if matching else None
        if not raised:
            if (
                audit_after != audit_before + 1
                or len(matching) != 1
                or receipt.get("outcome") != "returned"
            ):
                raise CleanupEvidenceError(
                    f"returned {category} adapter call lacks one exact receipt"
                )
            outcome = "returned"
        elif receipt is not None:
            if (
                audit_after != audit_before + 1
                or len(matching) != 1
                or receipt.get("outcome") != "raised"
            ):
                raise CleanupEvidenceError(
                    f"raised {category} adapter call has inconsistent evidence"
                )
            outcome = "raised"
        else:
            outcome = "uncertain"
        return request, outcome, receipt, audit_before, audit_after

    def send_exact_zero(
        self,
        command: Mapping[str, float],
        *,
        deadline_monotonic_ns: int,
    ) -> Mapping[str, Any]:
        from competition.adapter import AttitudeRateCommand

        checked = dict(command)
        if not powered_runtime.exact_zero_rate_thrust(checked):
            raise CleanupExecutionError("cleanup adapter rejected nonzero target")
        request, outcome, receipt, before, after = self._call_and_receipt(
            self.adapter.send_attitude_rate(
                AttitudeRateCommand(0.0, 0.0, 0.0, 0.0),
                powered_deadline_monotonic_ns=deadline_monotonic_ns,
                powered_cleanup=True,
            ),
            category="attitude_target",
            audit_name="attitude_target",
        )
        evidence = self.zero_evidence_factory(
            command=checked,
            request_monotonic_ns=request,
            completed_monotonic_ns=self._now(),
            outcome=outcome,
            receipt=receipt,
            audit_count_before=before,
            audit_count_after=after,
        )
        if not isinstance(evidence, Mapping):
            raise CleanupEvidenceError("cleanup zero evidence factory failed")
        return dict(evidence)

    def send_disarm(self, *, deadline_monotonic_ns: int) -> NonattitudeDispatch:
        request, outcome, receipt, _before, _after = self._call_and_receipt(
            self.adapter.disarm(
                powered_deadline_monotonic_ns=deadline_monotonic_ns,
                powered_cleanup=True,
            ),
            category="disarm",
            audit_name="disarm",
        )
        return NonattitudeDispatch(request, outcome, receipt)

    def send_reset(
        self,
        *,
        baseline: Mapping[str, Any],
        deadline_monotonic_ns: int,
    ) -> ResetDispatch:
        del baseline
        boundary: dict[str, Any] | None = None

        def retain(value: Any) -> None:
            nonlocal boundary
            if boundary is not None:
                raise CleanupEvidenceError("reset boundary callback was repeated")
            primitive = self._primitive(value)
            old_generation = primitive.get("old_generation")
            new_generation = primitive.get("new_generation")
            boundary_time = primitive.get("boundary_monotonic_ns")
            collisions = primitive.get("collisions")
            if type(collisions) is not list:
                raise CleanupEvidenceError(
                    "reset boundary collision payload is not an array"
                )
            primitive["collisions"] = [
                self._record_collision(
                    item,
                    generation=old_generation,
                    observed_monotonic_ns=boundary_time,
                    phase="fallback-reset-and-epoch",
                    disposition="reset_boundary_discard",
                )
                for item in collisions
            ]
            try:
                boundary = attempt_contract.validate_reset_boundary(primitive)
            except BaseException as exc:
                raise CleanupEvidenceError(
                    "normalized reset boundary validation failed"
                ) from exc
            self._begin_collision_generation(new_generation)

        request, outcome, receipt, _before, _after = self._call_and_receipt(
            self.adapter.reset_calibration_with_boundary(
                retain,
                powered_deadline_monotonic_ns=deadline_monotonic_ns,
                powered_cleanup=True,
            ),
            category="sim_reset",
            audit_name="sim_reset",
        )
        return ResetDispatch(
            NonattitudeDispatch(request, outcome, receipt),
            {} if boundary is None else boundary,
        )

    def outbound_receipts(self) -> Sequence[Mapping[str, Any]]:
        self._capture_receipts()
        return [dict(item) for item in self._receipts]

    def outbound_audit(self) -> Mapping[str, Any]:
        self._capture_receipts()
        audit = self.adapter.outbound_audit()
        categories = {
            name: getattr(audit, name)
            for name in (
                "timesync",
                "gcs_heartbeat",
                "sim_reset",
                "arm",
                "disarm",
                "attitude_target",
                "position_target",
                "other_command",
            )
        }
        returned = sum(item.get("outcome") == "returned" for item in self._receipts)
        raised = sum(item.get("outcome") == "raised" for item in self._receipts)
        attempted = sum(categories.values())
        if returned + raised > attempted:
            raise CleanupEvidenceError(
                "adapter receipt count exceeds attempted outbound calls"
            )
        return {
            **categories,
            "receipt_count": returned + raised,
            "receipt_returned": returned,
            "receipt_raised": raised,
            "receipt_dropped": attempted - returned - raised,
            "receipt_buffered": 0,
        }

    def collision_observations(self) -> Sequence[Mapping[str, Any]]:
        stats = self.adapter.collision_stats()
        generation = getattr(stats, "generation", None)
        for item in self.adapter.drain_collisions():
            self._record_collision(
                item,
                generation=generation,
                observed_monotonic_ns=self._now(),
                phase="fallback-finalize",
                disposition="cleanup_continue",
            )
        return [
            self._copy_collision_observation(item)
            for item in self._collision_observations
        ]

    def request_stop(self) -> None:
        self._stop_requested = True

    def join_workers(
        self,
        *,
        deadline_monotonic_ns: int,
        progress_callback: Callable[[], None] | None = None,
    ) -> WorkerCloseProof:
        self._run(
            self.adapter.disconnect(
                deadline_monotonic_ns=deadline_monotonic_ns,
                powered_progress=progress_callback,
            )
        )
        state = self.adapter.powered_transport_state()
        if state is None:
            raise CleanupEvidenceError(
                "powered adapter transport state is unavailable after close"
            )
        return WorkerCloseProof(
            receiver_joined=state.receiver_joined,
            announcer_joined=state.announcer_joined,
            owned_handles_closed=state.owned_handles_closed,
        )


def create_aigp_mavlink_cleanup_backend(
    endpoint: powered_runtime.ExclusiveUdpEndpoint,
    authority: CleanupBackendAuthority,
    *,
    zero_evidence_factory: Callable[..., Mapping[str, Any]],
    monotonic_ns: Callable[[], int] = powered_runtime.read_qpc_ns,
    transport_factory: Callable[..., Any] | None = None,
) -> AIGPMavlinkCleanupBackend:
    """Construct the production parser/adapter bridge without opening a port."""

    from competition.aigp_mavlink import (
        AIGPMavlinkAdapter,
        PoweredMavlinkTransport,
        POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
    )

    if not isinstance(authority, CleanupBackendAuthority):
        endpoint.close()
        raise TypeError("cleanup backend authority is invalid")
    build_transport = transport_factory or PoweredMavlinkTransport.from_pymavlink
    try:
        transport = build_transport(
            endpoint,
            outbound_guards=authority.outbound_guards,
            role_valid=authority.role_valid,
            parent_alive=authority.parent_alive,
            lease_valid=authority.lease_valid,
            external_cleanup_authorize=authority.authorize_outbound,
        )
        adapter = AIGPMavlinkAdapter(
            enable_vision=False,
            require_track=False,
            telemetry_mode="imu",
            fetch_track_on_connect=False,
            monotonic_ns=monotonic_ns,
            powered_transport=transport,
            powered_receive_mode=POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        )
        return AIGPMavlinkCleanupBackend(
            adapter,
            zero_evidence_factory=zero_evidence_factory,
            monotonic_ns=monotonic_ns,
        )
    except BaseException:
        if not endpoint.closed:
            endpoint.close()
        raise


@dataclass
class CleanupAdmission:
    arguments: CleanupArguments
    attempt: dict[str, Any]
    live_freeze: dict[str, Any]
    process_authority: dict[str, Any]
    current_process: dict[str, Any]
    wrapper_process: dict[str, Any]
    process_argv: tuple[str, ...]
    capability_handle: int
    parent_handle: int
    role_secret: bytearray = field(repr=False)
    admitted_monotonic_ns: int = 0
    total_deadline_monotonic_ns: int = 0
    attempt_envelope_sha256: str = ""
    process_authority_sha256: str = ""

    def erase_role_secret(self) -> None:
        for index in range(len(self.role_secret)):
            self.role_secret[index] = 0


@dataclass(frozen=True)
class CleanupRunOutput:
    certificate: dict[str, Any]
    certificate_sha256: str
    process_result: dict[str, Any]
    exit_code: int


def _stable_record(path: str, contract: Any) -> Mapping[str, Any]:
    before = powered_runtime.stable_file_identity(path)
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise CleanupBootstrapError("immutable bootstrap record could not be read") from exc
    after = powered_runtime.stable_file_identity(path)
    digest = hashlib.sha256(payload).hexdigest()
    if before != after or digest != before.sha256:
        raise CleanupBootstrapError("immutable bootstrap record changed while reading")
    try:
        return contract.parse_canonical_json_bytes(payload, file_form=True)
    except BaseException as exc:
        raise CleanupBootstrapError("bootstrap record is not canonical JSON") from exc


def _load_record(path: str, services: CleanupServices) -> Mapping[str, Any]:
    loader = services.load_record or _stable_record
    return loader(path, services.contract)


def _exact_mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CleanupBootstrapError(f"{label} must be an exact object")
    return dict(value)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CleanupBootstrapError(f"{label} does not match immutable authority")


def admit_cleanup_fallback(
    arguments: CleanupArguments,
    services: CleanupServices,
) -> CleanupAdmission:
    """Validate immutable authority and consume one role capability.

    No endpoint/backend factory is referenced before this function succeeds.
    """

    if not isinstance(arguments, CleanupArguments):
        raise TypeError("arguments must be CleanupArguments")
    contract = services.contract
    capability_handle = powered_runtime.parse_decimal_handle(
        arguments.cleanup_capability_handle
    )
    parent_handle = powered_runtime.parse_decimal_handle(
        arguments.parent_liveness_handle
    )
    if capability_handle == parent_handle:
        raise CleanupBootstrapError("capability and parent handles must be distinct")
    wrapper_token = powered_runtime.parse_process_identity_token(
        arguments.wrapper_process
    )

    try:
        attempt_initial = contract.validate_attempt(
            _load_record(arguments.powered_attempt_envelope, services)
        )
        live_freeze_path = attempt_initial["context"]["live_freeze"]["path"]
        live_freeze = contract.validate_live_freeze(
            _load_record(live_freeze_path, services)
        )
        attempt = contract.validate_attempt(attempt_initial, live_freeze=live_freeze)
        process_argv = tuple(services.process_boundary.current_argv())
        if not process_argv or any(type(item) is not str for item in process_argv):
            raise CleanupBootstrapError("current process argv proof is invalid")
        authority = contract.validate_process_authority(
            _load_record(arguments.powered_process_authority, services),
            attempt=attempt,
            argv=process_argv,
        )
    except CleanupFallbackError:
        raise
    except BaseException as exc:
        raise CleanupBootstrapError("attempt or process authority validation failed") from exc

    if authority["role"] != ROLE:
        raise CleanupBootstrapError("process authority role is not cleanup fallback")
    context = attempt["context"]
    paths = context["paths"]
    _require_equal(
        arguments.powered_attempt_envelope,
        paths["attempt_envelope"],
        "attempt-envelope path",
    )
    _require_equal(
        arguments.powered_process_authority,
        paths["cleanup_authority"],
        "process-authority path",
    )
    _require_equal(
        arguments.cleanup_certificate,
        paths["fallback_cleanup_certificate"],
        "cleanup-certificate path",
    )
    _require_equal(process_argv, tuple(context["cleanup_argv"]), "cleanup argv")
    _require_equal(parent_handle, authority["parent_handle"]["value"], "parent handle")
    wrapper_expected = _exact_mapping(context["wrapper_process"], "wrapper process")
    if (
        wrapper_token.pid != wrapper_expected["pid"]
        or wrapper_token.creation_filetime_100ns
        != wrapper_expected["creation_filetime_100ns"]
    ):
        raise CleanupBootstrapError("wrapper identity token does not match attempt")

    try:
        current_process = powered_runtime.validate_process_identity(
            services.process_boundary.current_process_identity()
        )
        retained_parent = powered_runtime.validate_process_identity(
            services.process_boundary.retained_process_identity(parent_handle)
        )
    except BaseException as exc:
        raise CleanupBootstrapError("retained process identity proof failed") from exc
    _require_equal(current_process, authority["process"], "current process identity")
    _require_equal(retained_parent, wrapper_expected, "retained wrapper identity")
    if services.process_boundary.prove_inherited_handle_policy(
        capability_handle=capability_handle,
        parent_handle=parent_handle,
        process_authority=authority,
    ) is not True:
        raise CleanupBootstrapError("inherited handle policy is unproved")

    anchor = authority["absolute_deadlines"]["anchor"]
    total_deadline = authority["absolute_deadlines"]["total"]
    capability_deadline = min(
        anchor + CAPABILITY_RELEASE_DURATION_NS,
        total_deadline,
    )
    try:
        secret = powered_runtime.read_bound_capability(
            capability_handle,
            parent_handle,
            domain=CAPABILITY_DOMAIN,
            context_sha256=attempt["context_sha256"],
            expected_capability_sha256=authority["capability_sha256"],
            deadline_monotonic_ns=capability_deadline,
            operations=services.capability_operations,
            monotonic_ns=services.monotonic_ns,
        )
    except BaseException as exc:
        raise CleanupBootstrapError("cleanup capability admission failed") from exc
    admitted = powered_runtime.read_qpc_ns(services.monotonic_ns)
    if admitted >= total_deadline:
        role_secret = bytearray(secret)
        for index in range(len(role_secret)):
            role_secret[index] = 0
        raise CleanupBootstrapError("fallback total deadline expired at admission")
    return CleanupAdmission(
        arguments=arguments,
        attempt=dict(attempt),
        live_freeze=dict(live_freeze),
        process_authority=dict(authority),
        current_process=current_process,
        wrapper_process=retained_parent,
        process_argv=process_argv,
        capability_handle=capability_handle,
        parent_handle=parent_handle,
        role_secret=bytearray(secret),
        admitted_monotonic_ns=admitted,
        total_deadline_monotonic_ns=total_deadline,
        attempt_envelope_sha256=contract.canonical_file_sha256(attempt),
        process_authority_sha256=contract.canonical_file_sha256(authority),
    )


@dataclass
class _ObservationState:
    heartbeat: dict[str, Any] | None = None
    race: dict[str, Any] | None = None
    imu: dict[str, Any] | None = None
    last_generation: int | None = None
    last_sequence: int | None = None

    def update(self, value: Mapping[str, Any], contract: Any) -> tuple[str, dict[str, Any]]:
        if type(value) is not dict or type(value.get("schema")) is not str:
            raise CleanupEvidenceError("production parser returned an invalid envelope")
        schema = value["schema"]
        try:
            if schema == _RECEIVED_HEARTBEAT:
                checked = contract.validate_received_heartbeat(value)
                stream = "HEARTBEAT"
            elif schema == _RECEIVED_RACE:
                checked = contract.validate_received_race_status(value)
                stream = "RACE_STATUS"
            elif schema == _RECEIVED_IMU:
                checked = contract.validate_received_imu(value)
                stream = "HIGHRES_IMU"
            else:
                raise CleanupEvidenceError(
                    "production parser returned an unsupported cleanup envelope"
                )
        except CleanupFallbackError:
            raise
        except BaseException as exc:
            raise CleanupEvidenceError(
                "production parser envelope validation failed"
            ) from exc
        ingress = checked["ingress"]
        generation = ingress["generation"]
        sequence = ingress["sequence"]
        if self.last_generation is not None:
            if generation < self.last_generation or generation > self.last_generation + 1:
                raise CleanupEvidenceError("received-envelope generation is discontinuous")
            if generation == self.last_generation and sequence <= self.last_sequence:
                raise CleanupEvidenceError("received-envelope sequence is not increasing")
        self.last_generation = generation
        self.last_sequence = sequence
        if stream == "HEARTBEAT":
            self.heartbeat = checked
        elif stream == "RACE_STATUS":
            self.race = checked
        else:
            self.imu = checked
        return stream, checked


def _not_attempted_zero() -> dict[str, Any]:
    return {
        "state": "not_attempted",
        "required": True,
        "requested": dict(ZERO_COMMAND),
        "generated": None,
        "terminal": None,
        "outbound_receipt": None,
    }


def _not_attempted_disarm() -> dict[str, Any]:
    return {
        "state": "not_attempted",
        "request_monotonic_ns": None,
        "receipt": None,
        "heartbeat_before": None,
        "heartbeat_after": None,
        "newer_confirmed": False,
    }


def _not_attempted_reset() -> dict[str, Any]:
    return {
        "state": "not_attempted",
        "request_monotonic_ns": None,
        "receipt": None,
        "boundary": None,
        "baseline": None,
        "clean_epoch": None,
        "advancing_race": [],
        "advancing_imu": [],
        "rollback_and_advance_confirmed": False,
    }


def _unobserved_final_state() -> dict[str, Any]:
    return {
        "state": "unobserved",
        "heartbeat": None,
        "disarmed": None,
        "reset_epoch": None,
        "last_race": None,
        "last_imu": None,
    }


def _zero_outbound_audit() -> dict[str, int]:
    return {
        "timesync": 0,
        "gcs_heartbeat": 0,
        "sim_reset": 0,
        "arm": 0,
        "disarm": 0,
        "attitude_target": 0,
        "position_target": 0,
        "other_command": 0,
        "receipt_count": 0,
        "receipt_returned": 0,
        "receipt_raised": 0,
        "receipt_dropped": 0,
        "receipt_buffered": 0,
    }


class CleanupFallbackMachine:
    """Single-use, cleanup-only synchronous fallback state machine."""

    def __init__(self, admission: CleanupAdmission, services: CleanupServices) -> None:
        self.admission = admission
        self.services = services
        self.contract = services.contract
        self.clock = services.monotonic_ns
        self.total_deadline = admission.total_deadline_monotonic_ns
        self.durations = admission.attempt["context"]["deadline_durations_ns"]
        self.phase_deadlines: list[dict[str, Any]] = []
        self.guard = powered_runtime.PoweredOutboundGuards()
        self.guard.latch_production("cleanup_fallback_has_no_production_authority")
        self.observations = _ObservationState()
        self.endpoint: powered_runtime.ExclusiveUdpEndpoint | None = None
        self.backend: CleanupMavlinkBackend | None = None
        self.source_gate: powered_runtime.MavlinkSourceFreeze | None = None
        self.lease_proof: LeaseProof | None = None
        self.parent_mode = "live_delegation"
        self.parent_observed_monotonic_ns = admission.admitted_monotonic_ns
        self.takeover_completed_monotonic_ns: int | None = None
        self.takeover_record_sha256: str | None = None
        self._parent_death_observed = False
        self.failure_codes: set[str] = set()
        self.collection_codes: set[str] = set()
        self.zero_command = _not_attempted_zero()
        self.disarm = _not_attempted_disarm()
        self.reset = _not_attempted_reset()
        self.final_state = _unobserved_final_state()
        self.worker_proof = WorkerCloseProof(False, False, False)
        self.socket_closed = False
        self._certificate_published = False
        self._cleanup_closed = False
        self._takeover_attempted = False
        self._active_phase_deadline: int | None = None
        self._active_phase: str | None = None
        self._takeover_heartbeat_due_ns: int | None = None
        self._takeover_heartbeat_hard_deadline_ns: int | None = None

    def _now(self) -> int:
        return powered_runtime.read_qpc_ns(self.clock)

    def _backend_parent_alive(self) -> bool:
        if self.parent_mode == "signaled_takeover":
            return False
        signaled = self.services.process_boundary.parent_signaled(
            self.admission.parent_handle
        )
        if type(signaled) is not bool:
            raise CleanupEvidenceError(
                "parent liveness state is not an exact boolean"
            )
        return not signaled

    def _phase(self, phase: str) -> powered_runtime.PhaseDeadline:
        duration_key = (
            "parent_death_lease_takeover"
            if phase == "parent_death_lease_takeover"
            else f"fallback_{phase}"
        )
        if duration_key not in self.durations:
            raise CleanupEvidenceError("fallback phase duration is unavailable")
        try:
            frozen = powered_runtime.freeze_phase_deadline(
                phase,
                self.durations[duration_key],
                self.total_deadline,
                monotonic_ns=self.clock,
            )
        except powered_runtime.PoweredDeadlineExpired as exc:
            self.failure_codes.add("deadline_expired")
            raise CleanupDeadlineError("fallback phase parent deadline expired") from exc
        row = frozen.to_primitive()
        self.phase_deadlines.append(row)
        self._active_phase_deadline = frozen.deadline_monotonic_ns
        self._active_phase = phase
        return frozen

    def _remaining_wait(self, deadline: int) -> int:
        now = self._now()
        if now >= deadline:
            self.failure_codes.add("deadline_expired")
            raise CleanupDeadlineError("fallback phase deadline expired")
        wait = min(powered_runtime.MAX_POLL_INTERVAL_NS, deadline - now)
        heartbeat_due = self._takeover_heartbeat_due_ns
        if heartbeat_due is not None and heartbeat_due > now:
            wait = min(wait, heartbeat_due - now)
        return max(1, wait)

    @staticmethod
    def _validate_lease_proof(
        proof: Any,
        *,
        owner_roles: frozenset[str],
    ) -> Any:
        try:
            owner_role = proof.owner_role
            generation = proof.generation
            record_sha256 = proof.record_sha256
            authority_valid = proof.authority_valid
            completed = proof.takeover_completed_monotonic_ns
        except (AttributeError, TypeError) as exc:
            raise CleanupEvidenceError("cleanup lease proof shape is invalid") from exc
        if type(owner_role) is not str or owner_role not in owner_roles:
            raise CleanupEvidenceError("cleanup lease proof owner is invalid")
        if type(generation) is not int or generation < 0:
            raise CleanupEvidenceError("cleanup lease proof generation is invalid")
        if (
            type(record_sha256) is not str
            or len(record_sha256) != 64
            or any(character not in "0123456789abcdef" for character in record_sha256)
        ):
            raise CleanupEvidenceError("cleanup lease proof hash is invalid")
        if type(authority_valid) is not bool:
            raise CleanupEvidenceError("cleanup lease proof authority is invalid")
        if completed is not None and (
            type(completed) is not int or completed < 0
        ):
            raise CleanupEvidenceError("cleanup lease completion is invalid")
        return proof

    def _invalidate_lease_proof(self) -> None:
        self._takeover_heartbeat_due_ns = None
        self._takeover_heartbeat_hard_deadline_ns = None
        proof = self.lease_proof
        if proof is None:
            self.lease_proof = LeaseProof(
                owner_role="wrapper",
                generation=0,
                record_sha256=self.admission.process_authority[
                    "lease_record_sha256"
                ],
                authority_valid=False,
            )
            return
        self.lease_proof = LeaseProof(
            owner_role=proof.owner_role,
            generation=proof.generation,
            record_sha256=proof.record_sha256,
            authority_valid=False,
            takeover_completed_monotonic_ns=proof.takeover_completed_monotonic_ns,
        )

    @staticmethod
    def _heartbeat_not_yet_due(exc: BaseException) -> bool:
        return bool(
            type(exc).__name__ == "LiveLeaseEvidenceError"
            and str(exc)
            == "delegated powered heartbeat preceded its frozen one-second cadence"
        )

    def _service_takeover_heartbeat(self, *, phase: str, deadline: int) -> None:
        if self.parent_mode != "signaled_takeover":
            return
        proof = self.lease_proof
        due = self._takeover_heartbeat_due_ns
        hard_deadline = self._takeover_heartbeat_hard_deadline_ns
        if proof is None or due is None or hard_deadline is None:
            self.failure_codes.add("lease_invalid")
            self._invalidate_lease_proof()
            raise CleanupExecutionError("takeover heartbeat state is incomplete")
        now = self._now()
        if now < due:
            return
        operation_deadline = min(deadline, hard_deadline, self.total_deadline)
        if now >= operation_deadline:
            self.failure_codes.update({"deadline_expired", "lease_invalid"})
            self._invalidate_lease_proof()
            raise CleanupDeadlineError("takeover heartbeat deadline expired")
        request_started = now
        try:
            refreshed = self.services.lease_boundary.heartbeat_takeover(
                proof,
                phase=phase,
                deadline_monotonic_ns=operation_deadline,
            )
        except BaseException as exc:
            if self._heartbeat_not_yet_due(exc):
                retry = request_started + powered_runtime.MAX_POLL_INTERVAL_NS
                if retry < operation_deadline:
                    self._takeover_heartbeat_due_ns = retry
                    return
            self.failure_codes.add("lease_invalid")
            self._invalidate_lease_proof()
            raise CleanupExecutionError("takeover heartbeat failed") from exc
        completed = self._now()
        if completed >= operation_deadline:
            self.failure_codes.update({"deadline_expired", "lease_invalid"})
            self._invalidate_lease_proof()
            raise CleanupDeadlineError("takeover heartbeat completed too late")
        try:
            self._validate_lease_proof(
                refreshed,
                owner_roles=frozenset({"cleanup-fallback-parent-death"}),
            )
        except BaseException:
            self.failure_codes.add("lease_invalid")
            self._invalidate_lease_proof()
            raise
        if (
            refreshed.authority_valid is not True
            or refreshed.generation != proof.generation + 1
            or refreshed.record_sha256 == proof.record_sha256
            or refreshed.takeover_completed_monotonic_ns
            != proof.takeover_completed_monotonic_ns
        ):
            self.failure_codes.add("lease_invalid")
            self._invalidate_lease_proof()
            raise CleanupEvidenceError("takeover heartbeat proof is stale")
        self.lease_proof = refreshed
        period = self.durations["lease_heartbeat_period"]
        maximum_gap = self.durations["lease_heartbeat_max_gap"]
        self._takeover_heartbeat_due_ns = request_started + period
        self._takeover_heartbeat_hard_deadline_ns = request_started + maximum_gap

    def _service_progress(self, *, phase: str, deadline: int) -> None:
        self._takeover_if_signaled()
        self._service_takeover_heartbeat(phase=phase, deadline=deadline)

    def _current_lease_phase(self, fallback: str) -> str:
        phase = self._active_phase or fallback
        return "fallback-" + phase.replace("_", "-")

    def _takeover_if_signaled(self) -> bool:
        signaled = self.services.process_boundary.parent_signaled(
            self.admission.parent_handle
        )
        if type(signaled) is not bool:
            raise CleanupEvidenceError("parent liveness state is not exact boolean")
        if not signaled:
            return False
        if self.parent_mode == "signaled_takeover":
            return True
        if self._takeover_attempted:
            self.failure_codes.update({"parent_dead", "lease_invalid"})
            self._invalidate_lease_proof()
            raise CleanupExecutionError("abandoned cleanup lease takeover was exhausted")
        self._takeover_attempted = True
        self._parent_death_observed = True
        self.failure_codes.add("parent_dead")
        self.parent_observed_monotonic_ns = self._now()
        self.guard.note_parent_death()
        phase = self._phase("parent_death_lease_takeover")
        takeover_started = self._now()
        try:
            proof = self.services.lease_boundary.take_over_abandoned(
                role_secret=memoryview(self.admission.role_secret),
                attempt=self.admission.attempt,
                process_authority=self.admission.process_authority,
                deadline_monotonic_ns=phase.deadline_monotonic_ns,
            )
        except BaseException as exc:
            self.failure_codes.update({"parent_dead", "lease_invalid"})
            self._invalidate_lease_proof()
            raise CleanupExecutionError("abandoned cleanup lease takeover failed") from exc
        try:
            self._validate_lease_proof(
                proof,
                owner_roles=frozenset({"cleanup-fallback-parent-death"}),
            )
        except BaseException:
            self.failure_codes.update({"parent_dead", "lease_invalid"})
            self._invalidate_lease_proof()
            raise
        if (
            proof.owner_role != "cleanup-fallback-parent-death"
            or proof.authority_valid is not True
            or proof.takeover_completed_monotonic_ns is None
            or proof.takeover_completed_monotonic_ns
            <= self.parent_observed_monotonic_ns
            or proof.takeover_completed_monotonic_ns
            >= phase.deadline_monotonic_ns
        ):
            self.failure_codes.update({"parent_dead", "lease_invalid"})
            self._invalidate_lease_proof()
            raise CleanupExecutionError("abandoned cleanup lease proof is invalid")
        self.lease_proof = proof
        self.parent_mode = "signaled_takeover"
        self.takeover_completed_monotonic_ns = proof.takeover_completed_monotonic_ns
        self.takeover_record_sha256 = proof.record_sha256
        self._takeover_heartbeat_due_ns = (
            takeover_started + self.durations["lease_heartbeat_period"]
        )
        self._takeover_heartbeat_hard_deadline_ns = (
            takeover_started + self.durations["lease_heartbeat_max_gap"]
        )
        if (
            not self._cleanup_closed
            and self.source_gate is not None
            and self.source_gate.promoted
        ):
            self.guard.enable_cleanup_takeover(
                parent_signaled=True,
                abandoned_lease_owned=True,
                authority_valid=True,
                source_promoted=True,
            )
        return True

    def _ensure_cleanup_guard(self) -> None:
        if self.lease_proof is None or self.lease_proof.authority_valid is not True:
            self.failure_codes.add("lease_invalid")
            raise CleanupExecutionError("cleanup lease authority is invalid")
        if self.source_gate is None or not self.source_gate.promoted:
            self.failure_codes.add("connect_failed")
            raise CleanupExecutionError("same-peer telemetry promotion is unproved")
        takeover = self._takeover_if_signaled()
        state = self.guard.cleanup_state
        if state == "disabled":
            if takeover:
                raise CleanupEvidenceError("takeover did not arm cleanup guard")
            self.guard.enable_cleanup_live(
                parent_alive=True,
                lease_valid=True,
                source_promoted=True,
            )
        elif state == "takeover_pending":
            if not takeover:
                raise CleanupEvidenceError("cleanup takeover is pending without parent death")
            self.guard.enable_cleanup_takeover(
                parent_signaled=True,
                abandoned_lease_owned=True,
                authority_valid=True,
                source_promoted=True,
            )

    def _authorize_send(
        self,
        category: str,
        *,
        phase_deadline: int,
        exact_zero: bool | None = None,
    ) -> int:
        self._ensure_cleanup_guard()
        self._service_takeover_heartbeat(
            phase=self._current_lease_phase("outbound"),
            deadline=phase_deadline,
        )
        now = self._now()
        call_deadline = min(
            now + self.durations["outbound_call"],
            phase_deadline,
            self.total_deadline,
        )
        if now >= call_deadline:
            self.failure_codes.add("deadline_expired")
            raise CleanupDeadlineError("outbound cleanup-call deadline expired")
        self.guard.authorize_cleanup(
            category,
            now_monotonic_ns=now,
            deadline_monotonic_ns=call_deadline,
            parent_alive=self.parent_mode == "live_delegation",
            lease_valid=self.lease_proof is not None
            and self.lease_proof.authority_valid,
            source_promoted=self.source_gate is not None
            and self.source_gate.promoted,
            exact_zero=exact_zero,
        )
        return call_deadline

    def authorize_backend_outbound(
        self,
        category: str,
        *,
        exact_zero: bool | None = None,
    ) -> int:
        """Dispatcher used by every backend-owned outbound call path."""

        if category in powered_runtime.ANNOUNCEMENT_CATEGORIES:
            if exact_zero is not None:
                raise powered_runtime.OutboundAuthorityError(
                    "announcement cannot carry exact-zero evidence"
                )
            if (
                self._cleanup_closed
                or self.guard.cleanup_state == "closed"
                or self.source_gate is None
                or not self.source_gate.outbound_permitted(category)
                or self.lease_proof is None
                or not self.lease_proof.authority_valid
                or self._active_phase_deadline is None
            ):
                raise powered_runtime.OutboundAuthorityError(
                    "cleanup announcement authority is unavailable"
                )
            self._service_progress(
                phase=self._current_lease_phase("announcement"),
                deadline=self._active_phase_deadline,
            )
            now = self._now()
            deadline = min(
                now + self.durations["outbound_call"],
                self._active_phase_deadline,
                self.total_deadline,
            )
            if now >= deadline:
                raise CleanupDeadlineError("cleanup announcement deadline expired")
            return deadline
        if self._active_phase_deadline is None:
            raise powered_runtime.OutboundAuthorityError(
                "cleanup phase authority is unavailable"
            )
        return self._authorize_send(
            category,
            phase_deadline=self._active_phase_deadline,
            exact_zero=exact_zero,
        )

    def _check_outbound_completion(self, deadline: int, failure_code: str) -> None:
        if self._now() >= deadline:
            self.failure_codes.update({"deadline_expired", failure_code})
            self.guard.close_cleanup()
            self._cleanup_closed = True
            raise CleanupDeadlineError("cleanup outbound call completed too late")

    def _consume_observations(
        self,
        *,
        accepted_source: bool,
        on_observation: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if self.backend is None or self.source_gate is None:
            raise CleanupEvidenceError("MAVLink backend is not constructed")
        rows = self.backend.drain_received_observations()
        if type(rows) not in {list, tuple}:
            raise CleanupEvidenceError("received observation drain must be a sequence")
        if rows and not accepted_source:
            raise CleanupEvidenceError(
                "production parser mutated state for an unaccepted datagram"
            )
        for item in rows:
            stream, checked = self.observations.update(item, self.contract)
            if on_observation is not None:
                on_observation(stream, checked)

    def _pump_until(
        self,
        predicate: Callable[[], bool],
        *,
        deadline: int,
        on_observation: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if self.backend is None or self.source_gate is None:
            raise CleanupEvidenceError("MAVLink backend is not constructed")
        while not predicate():
            lease_phase = self._current_lease_phase("poll")
            self._service_progress(phase=lease_phase, deadline=deadline)
            wait_ns = self._remaining_wait(deadline)
            dispatch = self.backend.receive_and_dispatch_datagram(wait_ns)
            self._service_progress(phase=lease_phase, deadline=deadline)
            if dispatch is None:
                self._consume_observations(
                    accepted_source=True,
                    on_observation=on_observation,
                )
                continue
            if not isinstance(dispatch, CleanupDatagramDispatch):
                raise CleanupEvidenceError("backend returned an invalid dispatch")
            if dispatch.failure_reason is not None:
                raise CleanupEvidenceError(
                    "adapter production datagram dispatch failed"
                )
            if dispatch.rejected_source:
                self.collection_codes.add("source_rejected")
                self._consume_observations(
                    accepted_source=False,
                    on_observation=on_observation,
                )
                continue
            if dispatch.malformed:
                self._consume_observations(
                    accepted_source=False,
                    on_observation=on_observation,
                )
                continue
            if not dispatch.source_accepted or not dispatch.production_dispatched:
                raise CleanupEvidenceError(
                    "accepted source did not reach production parsing"
                )
            self._consume_observations(
                accepted_source=True,
                on_observation=on_observation,
            )

    def _connect(self) -> None:
        phase = self._phase("connect")
        self._service_progress(
            phase="fallback-connect",
            deadline=phase.deadline_monotonic_ns,
        )
        bind = self.admission.live_freeze["transport"]["mavlink_bind"]
        self.endpoint = self.services.endpoint_factory(bind["host"], bind["port"])
        try:
            authority = CleanupBackendAuthority(
                outbound_guards=self.guard,
                authorize_outbound=self.authorize_backend_outbound,
                role_valid=lambda: False,
                parent_alive=self._backend_parent_alive,
                lease_valid=lambda: bool(
                    self.lease_proof is not None
                    and self.lease_proof.authority_valid
                ),
            )
            self.backend = self.services.backend_factory(
                self.endpoint,
                authority,
            )
            self.source_gate = self.backend.source_authority
            if type(self.source_gate) is not powered_runtime.MavlinkSourceFreeze:
                raise CleanupEvidenceError(
                    "backend did not expose the exact transport source authority"
                )
            self.backend.open(deadline_monotonic_ns=phase.deadline_monotonic_ns)
            self._service_progress(
                phase="fallback-connect",
                deadline=phase.deadline_monotonic_ns,
            )
            self._pump_until(
                lambda: bool(
                    self.source_gate is not None
                    and self.source_gate.promoted
                    and self.observations.heartbeat is not None
                    and self.observations.race is not None
                    and self.observations.imu is not None
                ),
                deadline=phase.deadline_monotonic_ns,
            )
            self._ensure_cleanup_guard()
        except BaseException:
            self.failure_codes.add("connect_failed")
            raise

    def _run_zero_and_disarm(self) -> None:
        if self.backend is None:
            raise CleanupEvidenceError("MAVLink backend is unavailable")
        phase = self._phase("disarm")
        zero_deadline = self._authorize_send(
            "attitude_target",
            phase_deadline=phase.deadline_monotonic_ns,
            exact_zero=True,
        )
        zero = self.backend.send_exact_zero(
            dict(ZERO_COMMAND),
            deadline_monotonic_ns=zero_deadline,
        )
        self._service_progress(
            phase="fallback-disarm",
            deadline=phase.deadline_monotonic_ns,
        )
        self._check_outbound_completion(zero_deadline, "zero_failed")
        if type(zero) is not dict:
            raise CleanupEvidenceError("zero-command evidence is not an exact object")
        self.zero_command = dict(zero)
        if (
            self.zero_command.get("state") != "returned"
            or self.zero_command.get("required") is not True
            or not powered_runtime.exact_zero_rate_thrust(
                self.zero_command.get("requested")
            )
        ):
            self.failure_codes.add("zero_failed")

        heartbeat_before = self.observations.heartbeat
        if heartbeat_before is None:
            self.failure_codes.add("disarm_failed")
            raise CleanupEvidenceError("disarm has no fresh heartbeat baseline")
        disarm_deadline = self._authorize_send(
            "disarm",
            phase_deadline=phase.deadline_monotonic_ns,
        )
        dispatch = self.backend.send_disarm(
            deadline_monotonic_ns=disarm_deadline
        )
        self._service_progress(
            phase="fallback-disarm",
            deadline=phase.deadline_monotonic_ns,
        )
        self._check_outbound_completion(disarm_deadline, "disarm_failed")
        if not isinstance(dispatch, NonattitudeDispatch):
            raise CleanupEvidenceError("disarm dispatch evidence is invalid")
        if dispatch.outcome != "returned":
            self.failure_codes.add("disarm_failed")
            self.disarm = {
                "state": "request_failed",
                "request_monotonic_ns": dispatch.request_monotonic_ns,
                "receipt": None if dispatch.outcome == "uncertain" else dispatch.receipt,
                "heartbeat_before": heartbeat_before,
                "heartbeat_after": None,
                "newer_confirmed": False,
            }
            return
        try:
            receipt = self.contract.validate_nonattitude_outbound(dispatch.receipt)
        except BaseException as exc:
            self.failure_codes.update({"disarm_failed", "receipt_incomplete"})
            raise CleanupEvidenceError("disarm receipt validation failed") from exc
        if receipt["category"] != "disarm" or receipt["outcome"] != "returned":
            self.failure_codes.update({"disarm_failed", "receipt_incomplete"})
            raise CleanupEvidenceError("disarm receipt is not a returned disarm call")
        before_sequence = heartbeat_before["ingress"]["sequence"]

        def confirmed() -> bool:
            current = self.observations.heartbeat
            return bool(
                current is not None
                and current["ingress"]["sequence"] > before_sequence
                and current["ingress"]["received_monotonic_ns"]
                > dispatch.request_monotonic_ns
                and current["heartbeat"]["base_mode"] & 128 == 0
            )

        try:
            self._pump_until(
                confirmed,
                deadline=phase.deadline_monotonic_ns,
            )
        except CleanupDeadlineError:
            self.failure_codes.add("disarm_failed")
        heartbeat_after = self.observations.heartbeat
        newer = confirmed()
        self.disarm = {
            "state": "confirmed" if newer else "unconfirmed",
            "request_monotonic_ns": dispatch.request_monotonic_ns,
            "receipt": receipt,
            "heartbeat_before": heartbeat_before,
            "heartbeat_after": heartbeat_after,
            "newer_confirmed": newer,
        }
        if not newer:
            self.failure_codes.add("disarm_failed")

    def _run_reset_and_epoch(self) -> None:
        if self.backend is None:
            raise CleanupEvidenceError("MAVLink backend is unavailable")
        phase = self._phase("reset_and_epoch")
        baseline_race = self.observations.race
        baseline_imu = self.observations.imu
        if baseline_race is None or baseline_imu is None:
            self.failure_codes.add("reset_failed")
            raise CleanupEvidenceError("reset has no fresh race/IMU baseline")
        baseline = {"race": baseline_race, "imu": baseline_imu}
        reset_deadline = self._authorize_send(
            "sim_reset",
            phase_deadline=phase.deadline_monotonic_ns,
        )
        dispatch = self.backend.send_reset(
            baseline=baseline,
            deadline_monotonic_ns=reset_deadline,
        )
        self._service_progress(
            phase="fallback-reset-and-epoch",
            deadline=phase.deadline_monotonic_ns,
        )
        self._check_outbound_completion(reset_deadline, "reset_failed")
        if not isinstance(dispatch, ResetDispatch):
            raise CleanupEvidenceError("reset dispatch evidence is invalid")
        if dispatch.request.outcome != "returned":
            self.failure_codes.add("reset_failed")
            self.reset = {
                "state": "request_failed",
                "request_monotonic_ns": dispatch.request.request_monotonic_ns,
                "receipt": (
                    None
                    if dispatch.request.outcome == "uncertain"
                    else dispatch.request.receipt
                ),
                "boundary": dict(dispatch.boundary),
                "baseline": baseline,
                "clean_epoch": None,
                "advancing_race": [],
                "advancing_imu": [],
                "rollback_and_advance_confirmed": False,
            }
            return
        try:
            boundary = self.contract.validate_reset_boundary(dispatch.boundary)
            receipt = self.contract.validate_nonattitude_outbound(
                dispatch.request.receipt
            )
        except BaseException as exc:
            self.failure_codes.update({"reset_failed", "receipt_incomplete"})
            raise CleanupEvidenceError("reset dispatch validation failed") from exc
        if receipt["category"] != "sim_reset" or receipt["outcome"] != "returned":
            self.failure_codes.update({"reset_failed", "receipt_incomplete"})
            raise CleanupEvidenceError("reset receipt is not a returned reset call")

        new_generation = boundary["new_generation"]
        race_anchor: int | None = None
        imu_anchor: int | None = None
        advancing_race: list[dict[str, Any]] = []
        advancing_imu: list[dict[str, Any]] = []
        post_reset_heartbeat: dict[str, Any] | None = None

        def collect(stream: str, row: dict[str, Any]) -> None:
            nonlocal race_anchor, imu_anchor, post_reset_heartbeat
            ingress = row["ingress"]
            if ingress["generation"] != new_generation:
                return
            if stream == "HEARTBEAT" and (
                ingress["received_monotonic_ns"]
                > dispatch.request.request_monotonic_ns
            ):
                post_reset_heartbeat = row
            elif stream == "RACE_STATUS":
                value = row["race_status"]["sim_boot_time_ms"]
                if race_anchor is None:
                    if value < baseline_race["race_status"]["sim_boot_time_ms"]:
                        race_anchor = value
                else:
                    advancing_race.append(row)
            elif stream == "HIGHRES_IMU":
                value = row["imu"]["timestamp_us"]
                if imu_anchor is None:
                    if value < baseline_imu["imu"]["timestamp_us"]:
                        imu_anchor = value
                else:
                    advancing_imu.append(row)

        def confirmed() -> bool:
            race_values = [
                row["race_status"]["sim_boot_time_ms"] for row in advancing_race
            ]
            imu_values = [row["imu"]["timestamp_us"] for row in advancing_imu]
            return bool(
                race_anchor is not None
                and imu_anchor is not None
                and len(advancing_race) >= 2
                and len(advancing_imu) >= 2
                and race_values[0] > race_anchor
                and imu_values[0] > imu_anchor
                and all(
                    later > earlier
                    for earlier, later in zip(race_values, race_values[1:])
                )
                and all(
                    later > earlier
                    for earlier, later in zip(imu_values, imu_values[1:])
                )
                and post_reset_heartbeat is not None
                and post_reset_heartbeat["heartbeat"]["base_mode"] & 128 == 0
            )

        try:
            self._pump_until(
                confirmed,
                deadline=phase.deadline_monotonic_ns,
                on_observation=collect,
            )
        except CleanupDeadlineError:
            self.failure_codes.add("reset_failed")
        proof = confirmed()
        clean_epoch = None
        if race_anchor is not None and imu_anchor is not None:
            clean_epoch = {
                "ingress_generation": new_generation,
                "race_anchor_boot_ms": race_anchor,
                "imu_anchor_usec": imu_anchor,
            }
        self.reset = {
            "state": "confirmed" if proof else "unconfirmed",
            "request_monotonic_ns": dispatch.request.request_monotonic_ns,
            "receipt": receipt,
            "boundary": boundary,
            "baseline": baseline,
            "clean_epoch": clean_epoch,
            "advancing_race": advancing_race,
            "advancing_imu": advancing_imu,
            "rollback_and_advance_confirmed": proof,
        }
        if proof:
            self.final_state = {
                "state": "confirmed",
                "heartbeat": post_reset_heartbeat,
                "disarmed": True,
                "reset_epoch": clean_epoch,
                "last_race": advancing_race[-1],
                "last_imu": advancing_imu[-1],
            }
        else:
            self.failure_codes.update({"reset_failed", "final_state_unproved"})

    def _close_transport(self, deadline: int) -> None:
        def progress() -> None:
            self._service_progress(phase="fallback-finalize", deadline=deadline)

        try:
            progress()
        except BaseException:
            self.failure_codes.add("lease_invalid")
        self.guard.close_cleanup()
        self._cleanup_closed = True
        if self.backend is None:
            self.worker_proof = WorkerCloseProof(True, True, True)
        else:
            try:
                self.backend.request_stop()
            except BaseException:
                self.failure_codes.add("transport_unclosed")
        if self.endpoint is None:
            self.socket_closed = True
        else:
            try:
                progress()
                self.endpoint.close()
                progress()
                self.socket_closed = self.endpoint.closed is True
            except BaseException:
                self.socket_closed = False
                self.failure_codes.add("transport_unclosed")
        if self.backend is not None:
            try:
                proof = self.backend.join_workers(
                    deadline_monotonic_ns=deadline,
                    progress_callback=progress,
                )
                if not isinstance(proof, WorkerCloseProof):
                    raise CleanupEvidenceError("worker close proof is invalid")
                self.worker_proof = proof
                if self._now() >= deadline:
                    raise CleanupDeadlineError("worker join completed too late")
            except BaseException:
                self.failure_codes.add("transport_unclosed")
        if not (
            self.socket_closed
            and self.worker_proof.receiver_joined
            and self.worker_proof.announcer_joined
            and self.worker_proof.owned_handles_closed
        ):
            self.failure_codes.add("transport_unclosed")

    def _emergency_close_no_wait(self) -> None:
        """Latch and close the socket even when no blocking deadline remains."""

        try:
            self.guard.close_cleanup()
            self._cleanup_closed = True
        except BaseException:
            pass
        if self.backend is not None:
            try:
                self.backend.request_stop()
            except BaseException:
                pass
        if self.endpoint is not None:
            try:
                self.endpoint.close()
                self.socket_closed = self.endpoint.closed is True
            except BaseException:
                self.socket_closed = False
        self.failure_codes.add("transport_unclosed")

    def _endpoint_evidence(self) -> dict[str, Any]:
        if self.endpoint is None:
            return {
                "state": "not_opened",
                "bind": None,
                "frozen_peer": None,
                "rejected_source_count": 0,
            }
        peer = None if self.source_gate is None else self.source_gate.peer
        if self.socket_closed:
            state = "closed_with_peer" if peer is not None else "closed_without_peer"
        else:
            state = "peer_frozen" if peer is not None else "bound"
        return {
            "state": state,
            "bind": {
                "role": "mavlink",
                "family": "AF_INET",
                "requested": {
                    "host": self.endpoint.requested_host,
                    "port": self.endpoint.requested_port,
                },
                "actual": {
                    "host": self.endpoint.actual_host,
                    "port": self.endpoint.actual_port,
                },
                "socket_policy": "ipv4-exclusive-address-use",
                "owner_process": self.admission.current_process,
            },
            "frozen_peer": (
                None if peer is None else {"host": peer[0], "port": peer[1]}
            ),
            "rejected_source_count": (
                0
                if self.source_gate is None
                else self.source_gate.rejected_source_count
            ),
        }

    def _collect_backend_evidence(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        if self.backend is None:
            return [], _zero_outbound_audit(), []
        deadline = self._active_phase_deadline or self.total_deadline

        def progress() -> None:
            try:
                self._service_progress(
                    phase="fallback-finalize",
                    deadline=deadline,
                )
            except BaseException:
                self.failure_codes.add("lease_invalid")

        try:
            progress()
            receipts_raw = self.backend.outbound_receipts()
            progress()
            audit_raw = self.backend.outbound_audit()
            progress()
            collisions_raw = self.backend.collision_observations()
            progress()
            if type(receipts_raw) not in {list, tuple}:
                raise CleanupEvidenceError("outbound receipts must be a sequence")
            if type(audit_raw) is not dict:
                raise CleanupEvidenceError("outbound audit must be an exact object")
            if type(collisions_raw) not in {list, tuple}:
                raise CleanupEvidenceError("collision observations must be a sequence")
            receipts = [dict(item) for item in receipts_raw]
            audit = dict(audit_raw)
            collisions = [dict(item) for item in collisions_raw]
        except BaseException as exc:
            self.failure_codes.update({"receipt_incomplete", "internal_error"})
            raise CleanupEvidenceError("backend evidence snapshot failed") from exc
        if collisions:
            self.collection_codes.add("collision_observed")
        for forbidden in ("arm", "position_target", "other_command"):
            if audit.get(forbidden) != 0:
                self.collection_codes.add("unexpected_outbound")
        if audit.get("receipt_dropped") != 0 or audit.get("receipt_buffered") != 0:
            self.failure_codes.add("receipt_incomplete")
        return receipts, audit, collisions

    def _cleanup_proved(self) -> bool:
        return bool(
            not self.failure_codes
            and self.lease_proof is not None
            and self.lease_proof.authority_valid
            and self._endpoint_evidence()["state"] == "closed_with_peer"
            and self.zero_command.get("state") == "returned"
            and self.disarm.get("state") == "confirmed"
            and self.reset.get("state") == "confirmed"
            and self.final_state.get("state") == "confirmed"
            and self.socket_closed
            and self.worker_proof.receiver_joined
            and self.worker_proof.announcer_joined
            and self.worker_proof.owned_handles_closed
            and self.guard.production_latched
            and self.guard.cleanup_state == "closed"
        )

    def _certificate(
        self,
        *,
        completed_monotonic_ns: int,
        receipts: list[dict[str, Any]],
        audit: dict[str, Any],
        collisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del audit  # Bound by process result; certificate binds complete receipts.
        proof = self.lease_proof
        if proof is None:
            proof = LeaseProof(
                owner_role="wrapper",
                generation=0,
                record_sha256=self.admission.process_authority["lease_record_sha256"],
                authority_valid=False,
            )
        outcome = "proved" if self._cleanup_proved() else "failed"
        if outcome == "failed" and not self.failure_codes:
            self.failure_codes.add("internal_error")
        certificate = {
            "schema": "aigp-vq2-powered-cleanup-certificate/1",
            "task_id": self.contract.TASK_ID,
            "session_id": self.contract.SESSION_ID,
            "attempt_id": self.contract.ATTEMPT_ID,
            "producer_role": ROLE,
            "cleanup_epoch": CLEANUP_EPOCH,
            "authority": {
                "process_authority": {
                    "path": self.admission.arguments.powered_process_authority,
                    "sha256": self.admission.process_authority_sha256,
                },
                "attempt_context_sha256": self.admission.attempt["context_sha256"],
                "attempt_envelope_sha256": self.admission.attempt_envelope_sha256,
                "producer": self.admission.current_process,
            },
            "trigger": (
                "parent_death"
                if self._parent_death_observed
                else "wrapper_fallback"
            ),
            "started_monotonic_ns": self.admission.admitted_monotonic_ns,
            "deadline_monotonic_ns": self.total_deadline,
            "completed_monotonic_ns": completed_monotonic_ns,
            "parent_state": {
                "mode": self.parent_mode,
                "wrapper_process": self.admission.wrapper_process,
                "observed_monotonic_ns": self.parent_observed_monotonic_ns,
                "takeover_completed_monotonic_ns": self.takeover_completed_monotonic_ns,
                "takeover_lease_record_sha256": self.takeover_record_sha256,
            },
            "lease": {
                "owner_role": proof.owner_role,
                "generation": proof.generation,
                "record_sha256": proof.record_sha256,
                "authority_valid": proof.authority_valid,
            },
            "phase_deadlines": list(self.phase_deadlines),
            "endpoints": {
                "mavlink": self._endpoint_evidence(),
                "camera": None,
            },
            "outbound_receipts": receipts,
            "zero_command": self.zero_command,
            "disarm": self.disarm,
            "reset": self.reset,
            "collisions": {
                "observations": collisions,
                "invalidating_occurrence_count": len(collisions),
            },
            "final_state": self.final_state,
            "transport": {
                "production_guard_latched": self.guard.production_latched,
                "cleanup_guard_closed": self.guard.cleanup_state == "closed",
                "vision_closed": True,
                "mavlink_socket_closed": self.socket_closed,
                "receiver_joined": self.worker_proof.receiver_joined,
                "announcer_joined": self.worker_proof.announcer_joined,
                "owned_handles_closed": self.worker_proof.owned_handles_closed,
            },
            "outcome": outcome,
            "failure_codes": sorted(self.failure_codes),
            "collection_invalidating_codes": sorted(self.collection_codes),
        }
        return certificate

    def _publish_certificate(
        self,
        certificate: Mapping[str, Any],
        *,
        deadline: int,
    ) -> tuple[dict[str, Any], str]:
        if self._certificate_published:
            raise CleanupEvidenceError("cleanup certificate publication was repeated")
        self._certificate_published = True

        def progress() -> None:
            try:
                self._service_progress(
                    phase="fallback-certificate",
                    deadline=deadline,
                )
            except BaseException:
                self.failure_codes.add("lease_invalid")

        progress()
        try:
            validated = self.contract.validate_cleanup_certificate(certificate)
        except BaseException as exc:
            raise CleanupEvidenceError("cleanup certificate validation failed") from exc
        digest = self.services.publisher.publish_create_new(
            self.admission.arguments.cleanup_certificate,
            validated,
            deadline_monotonic_ns=deadline,
            progress_callback=progress,
        )
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CleanupEvidenceError("cleanup certificate publication hash is invalid")
        expected = self.contract.canonical_file_sha256(validated)
        if digest != expected:
            raise CleanupEvidenceError("cleanup certificate readback hash mismatched")
        return dict(validated), digest

    def _release_takeover(self) -> bool:
        if self.parent_mode != "signaled_takeover":
            return True
        if self.lease_proof is None:
            return False
        now = self._now()
        deadline = min(
            now + self.durations["lease_release_and_verify"],
            self.total_deadline,
        )
        if now >= deadline:
            return False
        try:
            self._service_takeover_heartbeat(
                phase="fallback-release",
                deadline=deadline,
            )
            return self.services.lease_boundary.release_takeover(
                self.lease_proof,
                deadline_monotonic_ns=deadline,
            ) is True
        except BaseException:
            return False

    def _result_reason_codes(
        self,
        certificate: Mapping[str, Any],
        *,
        takeover_released: bool,
    ) -> list[str]:
        reasons: set[str] = set()
        if certificate["outcome"] != "proved":
            reasons.add("cleanup_unconfirmed")
        if "deadline_expired" in self.failure_codes:
            reasons.add("deadline_expired")
        if "internal_error" in self.failure_codes:
            reasons.add("internal_error")
        if self._parent_death_observed:
            reasons.add("wrapper_death")
            if self.parent_mode == "signaled_takeover" and not takeover_released:
                reasons.add("lease_release_unconfirmed")
        if "unexpected_outbound" in self.collection_codes:
            reasons.add("unexpected_outbound")
        if self.collection_codes - {"unexpected_outbound"}:
            reasons.add("capture_incomplete")
        return sorted(reasons)

    def _process_result(
        self,
        certificate: Mapping[str, Any],
        certificate_sha256: str,
        audit: Mapping[str, Any],
        *,
        takeover_released: bool,
    ) -> dict[str, Any]:
        reasons = self._result_reason_codes(
            certificate,
            takeover_released=takeover_released,
        )
        result = {
            "schema": "aigp-vq2-powered-process-result/1",
            "task_id": self.contract.TASK_ID,
            "session_id": self.contract.SESSION_ID,
            "attempt_id": self.contract.ATTEMPT_ID,
            "producer_role": ROLE,
            "process_authority_sha256": self.admission.process_authority_sha256,
            "started_monotonic_ns": self.admission.process_authority[
                "absolute_deadlines"
            ]["anchor"],
            "completed_monotonic_ns": self._now(),
            "outcome": "completed" if not reasons else "failed",
            "reason_codes": reasons,
            "phase_deadlines": list(self.phase_deadlines),
            "cleanup_certificate": {
                "path": self.admission.arguments.cleanup_certificate,
                "state": "published",
                "sha256": certificate_sha256,
            },
            "outbound_audit": dict(audit),
            "artifacts": {"legacy_record": None, "replay_bundle": None},
        }
        try:
            return self.contract.validate_process_result(
                result,
                cleanup_certificate=certificate,
            )
        except BaseException as exc:
            raise CleanupEvidenceError("cleanup process result validation failed") from exc

    def run(self) -> CleanupRunOutput:
        """Run once, publish one certificate, and return one process result."""

        try:
            try:
                boundary = self.services.lease_boundary
                if boundary is None:
                    raise CleanupEvidenceError("cleanup lease boundary is unavailable")
                proof = boundary.prove_live_delegation(
                    attempt=self.admission.attempt,
                    process_authority=self.admission.process_authority,
                )
                try:
                    self._validate_lease_proof(
                        proof,
                        owner_roles=frozenset({"wrapper"}),
                    )
                except BaseException:
                    self.failure_codes.update(
                        {"authority_invalid", "lease_invalid"}
                    )
                    raise
                self.lease_proof = proof
                if proof.owner_role != "wrapper" or not proof.authority_valid:
                    self.failure_codes.update({"authority_invalid", "lease_invalid"})
                else:
                    self._takeover_if_signaled()
                    self._connect()
            except BaseException:
                if not self.failure_codes:
                    self.failure_codes.add("connect_failed")

            if (
                self.backend is not None
                and self.source_gate is not None
                and self.source_gate.promoted
                and self.lease_proof is not None
                and self.lease_proof.authority_valid
            ):
                try:
                    self._run_zero_and_disarm()
                except BaseException:
                    if not ({"zero_failed", "disarm_failed"} & self.failure_codes):
                        self.failure_codes.add("internal_error")
                try:
                    self._run_reset_and_epoch()
                except BaseException:
                    if "reset_failed" not in self.failure_codes:
                        self.failure_codes.add("internal_error")

            try:
                self._takeover_if_signaled()
            except BaseException:
                self.failure_codes.update({"parent_dead", "lease_invalid"})
            try:
                finalize = self._phase("finalize")
            except BaseException:
                self._emergency_close_no_wait()
                raise CleanupDeadlineError("fallback finalize phase could not start")
            self._close_transport(finalize.deadline_monotonic_ns)
            receipts, audit, collisions = self._collect_backend_evidence()
            completed = self._now()
            if completed >= finalize.deadline_monotonic_ns:
                self.failure_codes.add("deadline_expired")
                raise CleanupDeadlineError("cleanup certificate missed finalize deadline")
            certificate = self._certificate(
                completed_monotonic_ns=completed,
                receipts=receipts,
                audit=audit,
                collisions=collisions,
            )
            certificate, certificate_sha256 = self._publish_certificate(
                certificate,
                deadline=finalize.deadline_monotonic_ns,
            )

            # One final liveness observation permits the contract's late,
            # cleanup-preserving takeover.  It never repeats zero/disarm/reset
            # and its phase therefore appears only in the process result.
            try:
                self._takeover_if_signaled()
            except BaseException:
                self.failure_codes.update({"parent_dead", "lease_invalid"})
            takeover_released = self._release_takeover()
            result = self._process_result(
                certificate,
                certificate_sha256,
                audit,
                takeover_released=takeover_released,
            )
            return CleanupRunOutput(
                certificate=certificate,
                certificate_sha256=certificate_sha256,
                process_result=result,
                exit_code=0 if result["outcome"] == "completed" else 1,
            )
        finally:
            self.admission.erase_role_secret()


class CanonicalCreateNewPublisher:
    """Create-new, flush, readback, and hash one canonical JSON artifact."""

    def __init__(
        self,
        *,
        contract: Any = attempt_contract,
        monotonic_ns: Callable[[], int] = powered_runtime.read_qpc_ns,
    ) -> None:
        self.contract = contract
        self.monotonic_ns = monotonic_ns

    def publish_create_new(
        self,
        path: str,
        value: Mapping[str, Any],
        *,
        deadline_monotonic_ns: int,
        progress_callback: Callable[[], None] | None = None,
    ) -> str:
        if type(path) is not str:
            raise CleanupEvidenceError("certificate path must be an exact string")
        target = Path(path)
        if not target.is_absolute() or os.path.normpath(path) != path:
            raise CleanupEvidenceError("certificate path must be canonical absolute")
        if progress_callback is not None:
            if not callable(progress_callback):
                raise TypeError("certificate progress callback must be callable")
            progress_callback()
        now = powered_runtime.read_qpc_ns(self.monotonic_ns)
        if now >= deadline_monotonic_ns:
            raise CleanupDeadlineError("certificate publication deadline expired")
        payload = self.contract.canonical_json_file_bytes(value)
        try:
            with target.open("xb") as stream:
                if progress_callback is not None:
                    progress_callback()
                stream.write(payload)
                stream.flush()
                if progress_callback is not None:
                    progress_callback()
                os.fsync(stream.fileno())
                if progress_callback is not None:
                    progress_callback()
        except OSError as exc:
            raise CleanupEvidenceError("create-new certificate publication failed") from exc
        if progress_callback is not None:
            progress_callback()
        identity = powered_runtime.stable_file_identity(target)
        if progress_callback is not None:
            progress_callback()
        if identity.size_bytes != len(payload) or identity.sha256 != hashlib.sha256(
            payload
        ).hexdigest():
            raise CleanupEvidenceError("certificate readback proof failed")
        if powered_runtime.read_qpc_ns(self.monotonic_ns) >= deadline_monotonic_ns:
            raise CleanupDeadlineError("certificate publication completed too late")
        return identity.sha256


def _cleanup_zero_evidence_factory(
    attempt: Mapping[str, Any],
    contract: Any,
) -> Callable[..., Mapping[str, Any]]:
    context = attempt["context"]
    common = {
        "attempt_id": context["attempt_id"],
        "session_id": context["session_id"],
        "candidate_commit": context["candidate_commit"],
        "attempt_context_sha256": attempt["context_sha256"],
        "host_clock_id": context["host"]["host_clock_id"],
        "reset_epoch": None,
        "plan": None,
        "scope": "cleanup_zero",
        "command_id": "cleanup/zero/0",
        "absolute_tick": None,
        "segment_id": None,
        "slot": None,
        "command": dict(ZERO_COMMAND),
        "source": {
            "frame": None,
            "imu": None,
            "race": None,
            "heartbeat": None,
            "actuator": None,
        },
    }

    def build(**values: Any) -> Mapping[str, Any]:
        command = values.get("command")
        request = values.get("request_monotonic_ns")
        completed = values.get("completed_monotonic_ns")
        outcome = values.get("outcome")
        receipt = values.get("receipt")
        before = values.get("audit_count_before")
        after = values.get("audit_count_after")
        if not powered_runtime.exact_zero_rate_thrust(command):
            raise CleanupEvidenceError("production zero evidence is not exact zero")
        for name, value in (
            ("zero request", request),
            ("zero completion", completed),
            ("zero audit before", before),
            ("zero audit after", after),
        ):
            if type(value) is not int or value < 0:
                raise CleanupEvidenceError(f"{name} is invalid")
        if completed < request or outcome not in {"returned", "raised", "uncertain"}:
            raise CleanupEvidenceError("production zero terminal state is invalid")
        watchdogs = {
            "checked_monotonic_ns": request,
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
        }
        generated = {
            **common,
            "watchdogs": watchdogs,
            "schema": "aigp-vq2-calibration-command-generated/1",
            "event_sequence": 0,
            "generated_monotonic_ns": request,
        }
        generated = contract.validate_command_generated(generated)
        generation_sha256 = contract.canonical_object_sha256(generated)
        terminal_common = {
            **common,
            "watchdogs": watchdogs,
            "event_sequence": 1,
            "generated_event_sequence": 0,
            "generation_sha256": generation_sha256,
        }
        checked_receipt = None if receipt is None else dict(receipt)
        if outcome == "returned":
            if checked_receipt is None:
                raise CleanupEvidenceError("returned zero has no outbound receipt")
            terminal = contract.validate_command_sent(
                {
                    **terminal_common,
                    "schema": "aigp-vq2-calibration-command-sent/1",
                    "sent_monotonic_ns": completed,
                    "transport": {
                        "receipt": checked_receipt,
                        "audit_count_before": before,
                        "audit_count_after": after,
                    },
                },
                generated=generated,
            )
            state = "returned"
        else:
            call_started = request if after == before + 1 else None
            call_ended = completed if checked_receipt is not None else None
            reason = "send_raised" if checked_receipt is not None else "internal_error"
            terminal = contract.validate_command_not_sent(
                {
                    **terminal_common,
                    "schema": "aigp-vq2-calibration-command-not-sent/1",
                    "recorded_monotonic_ns": completed,
                    "outcome": {
                        "kind": "send_failed_or_uncertain",
                        "reason_code": reason,
                        "detail": "cleanup zero adapter call did not return",
                        "audit_count_before": before,
                        "audit_count_after": after,
                        "call_started_monotonic_ns": call_started,
                        "call_ended_monotonic_ns": call_ended,
                    },
                },
                generated=generated,
            )
            state = "failed"
        return {
            "state": state,
            "required": True,
            "requested": dict(command),
            "generated": generated,
            "terminal": terminal,
            "outbound_receipt": checked_receipt,
        }

    return build


def _create_default_delegated_lease_boundary(
    admission: CleanupAdmission,
    process_boundary: ProcessBoundary,
    qpc_provider: Any,
) -> LeaseBoundary:
    # This import and all lease construction occur only after capability
    # admission. The delegated boundary opens the production mutex only if a
    # retained parent handle is subsequently observed signaled.
    from scripts import aigp_live_lease

    context = admission.attempt["context"]
    frequency = qpc_provider.query_performance_frequency_hz()
    if frequency != context["host"]["qpc_frequency_hz"]:
        raise CleanupBootstrapError("runtime QPC frequency changed from admission")
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


def _close_owned_process_boundary(
    process_boundary: Any,
    monotonic_ns: Callable[[], int],
) -> bool:
    started = powered_runtime.read_qpc_ns(monotonic_ns)
    deadline = started + OWNED_HANDLE_CLOSE_DURATION_NS
    try:
        proof = process_boundary.close_owned_handles(
            deadline_monotonic_ns=deadline,
            monotonic_ns=monotonic_ns,
        )
    except BaseException:
        return False
    return bool(getattr(proof, "proved", False))


def build_default_cleanup_services(
    arguments: CleanupArguments,
    *,
    qpc_provider_factory: Callable[[], Any] | None = None,
    process_boundary_factory: Callable[..., ProcessBoundary] | None = None,
    capability_operations_factory: Callable[[], Any] | None = None,
    delegated_lease_factory: Callable[
        [CleanupAdmission, ProcessBoundary, Any], LeaseBoundary
    ]
    | None = None,
    backend_builder: Callable[..., CleanupMavlinkBackend] | None = None,
    publisher_factory: Callable[..., CertificatePublisher] | None = None,
) -> CleanupServices:
    """Build the inert bootstrap surface for the production fallback.

    The returned service defers both the delegated lease import/construction
    and the MAVLink adapter import/construction until after the one-shot role
    capability has been consumed successfully.
    """

    if not isinstance(arguments, CleanupArguments):
        raise TypeError("arguments must be CleanupArguments")
    capability_handle = powered_runtime.parse_decimal_handle(
        arguments.cleanup_capability_handle
    )
    parent_handle = powered_runtime.parse_decimal_handle(
        arguments.parent_liveness_handle
    )
    make_qpc = qpc_provider_factory or powered_runtime.WindowsQpcProvider
    make_process = (
        process_boundary_factory
        or powered_runtime.RetainedChildBootstrapProcessBoundary
    )
    make_capability = (
        capability_operations_factory
        or powered_runtime.Win32CapabilityPipeOperations
    )
    make_lease = delegated_lease_factory or _create_default_delegated_lease_boundary
    make_backend = backend_builder or create_aigp_mavlink_cleanup_backend
    make_publisher = publisher_factory or CanonicalCreateNewPublisher
    qpc = make_qpc()
    process_boundary: ProcessBoundary | None = None
    try:
        capability_operations = make_capability()
        process_boundary = make_process(capability_handle, parent_handle)
        admission_holder: dict[str, CleanupAdmission] = {}

        def lease_after_capability(admission: CleanupAdmission) -> LeaseBoundary:
            admission_holder["value"] = admission
            return make_lease(admission, process_boundary, qpc)

        def backend_after_capability(
            endpoint: powered_runtime.ExclusiveUdpEndpoint,
            authority: CleanupBackendAuthority,
        ) -> CleanupMavlinkBackend:
            admission = admission_holder.get("value")
            if admission is None:
                endpoint.close()
                raise CleanupBootstrapError(
                    "MAVLink backend was requested before capability admission"
                )
            return make_backend(
                endpoint,
                authority,
                zero_evidence_factory=_cleanup_zero_evidence_factory(
                    admission.attempt,
                    attempt_contract,
                ),
                monotonic_ns=qpc.now_ns,
            )

        publisher = make_publisher(
            contract=attempt_contract,
            monotonic_ns=qpc.now_ns,
        )
        return CleanupServices(
            process_boundary=process_boundary,
            lease_boundary=None,
            backend_factory=backend_after_capability,
            publisher=publisher,
            capability_operations=capability_operations,
            monotonic_ns=qpc.now_ns,
            contract=attempt_contract,
            lease_boundary_factory=lease_after_capability,
            owned_process_boundary=process_boundary,
        )
    except BaseException:
        if process_boundary is not None:
            _close_owned_process_boundary(process_boundary, qpc.now_ns)
        raise


def run_cleanup_fallback(
    arguments: CleanupArguments,
    services: CleanupServices,
) -> CleanupRunOutput:
    admission = admit_cleanup_fallback(arguments, services)
    try:
        if services.lease_boundary is None:
            factory = services.lease_boundary_factory
            if factory is None:
                raise CleanupBootstrapError(
                    "post-capability lease boundary is unavailable"
                )
            services.lease_boundary = factory(admission)
            if services.lease_boundary is None:
                raise CleanupBootstrapError(
                    "post-capability lease construction failed"
                )
        return CleanupFallbackMachine(admission, services).run()
    except BaseException:
        admission.erase_role_secret()
        raise


def _write_sanitized_stderr(stream: Any, code: str) -> None:
    safe_codes = {
        "bootstrap": b"powered cleanup failed before admission\n",
        "execution": b"powered cleanup failed after admission\n",
    }
    payload = safe_codes.get(code, b"powered cleanup failed\n")
    if len(payload) > STDERR_LIMIT_BYTES:
        payload = payload[:STDERR_LIMIT_BYTES]
    stream.write(payload)
    if hasattr(stream, "flush"):
        stream.flush()


def main(
    argv: Sequence[str] | None = None,
    *,
    services: CleanupServices | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    output_stream = sys.stdout.buffer if stdout is None else stdout
    error_stream = sys.stderr.buffer if stderr is None else stderr
    parsed = parse_cleanup_arguments(args)
    active_services = services
    output: CleanupRunOutput | None = None
    failure_kind: str | None = None
    failure_code = 1
    try:
        if active_services is None:
            active_services = build_default_cleanup_services(parsed)
        output = run_cleanup_fallback(parsed, active_services)
    except CleanupBootstrapError:
        failure_kind = "bootstrap"
        failure_code = 2
    except BaseException:
        failure_kind = "execution"
        failure_code = 1
    if (
        active_services is not None
        and active_services.owned_process_boundary is not None
        and not _close_owned_process_boundary(
            active_services.owned_process_boundary,
            active_services.monotonic_ns,
        )
    ):
        failure_kind = "execution"
        failure_code = 1
        output = None
    if failure_kind is not None or output is None or active_services is None:
        _write_sanitized_stderr(
            error_stream,
            "execution" if failure_kind is None else failure_kind,
        )
        return failure_code
    payload = active_services.contract.canonical_json_file_bytes(
        output.process_result
    )
    output_stream.write(payload)
    if hasattr(output_stream, "flush"):
        output_stream.flush()
    return output.exit_code


if __name__ == "__main__":  # pragma: no cover - production entry point
    raise SystemExit(main())
