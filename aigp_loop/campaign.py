"""Plan-only warm-simulator campaign scaffold.

No powered executor is shipped: a caller-provided watchdog declaration cannot
prove that the watchdog code owns the simulator power boundary or can stop a
hung backend.  ``WarmCampaign.run`` therefore fails closed until a separately
reviewed, pinned out-of-process supervisor is integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from ._util import canonical_json, json_hash
from .ledger import TrialLedger


SIMULATOR_BUILD = "3385"
POWERED_STAGES = frozenset({"sign-id", "hover", "gate0", "gate0-observe"})


@dataclass(frozen=True)
class CampaignCandidate:
    trial_id: str
    label: str
    stage: str
    code_hash: str
    config_hash: str
    dataset_hash: str
    evaluator_version: str
    is_baseline: bool = False

    def __post_init__(self) -> None:
        for name in (
            "trial_id",
            "label",
            "stage",
            "evaluator_version",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(
                    f"campaign candidate {name} must be an exact non-empty string"
                )
        for name in ("code_hash", "config_hash", "dataset_hash"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef" for character in value
                )
            ):
                raise ValueError(
                    f"campaign candidate {name} must be a lowercase SHA-256"
                )
        if type(self.is_baseline) is not bool:
            raise TypeError("is_baseline must be an exact bool")


@dataclass(frozen=True)
class PreflightHealth:
    passed: bool
    simulator_build: str
    process_uptime_s: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("preflight passed must be an exact bool")
        if type(self.simulator_build) is not str or not self.simulator_build.strip():
            raise ValueError("simulator_build must be non-empty")
        if (
            type(self.process_uptime_s) not in {int, float}
            or not math.isfinite(self.process_uptime_s)
            or self.process_uptime_s < 0.0
        ):
            raise ValueError("process_uptime_s must be finite and non-negative")
        if not isinstance(self.details, Mapping):
            raise TypeError("preflight details must be an object")
        json_hash(self.details)


@dataclass(frozen=True)
class PoweredTrialResult:
    success: bool
    reset_epoch_proved: bool
    countdown_go_observed: bool
    watchdogs_armed: bool
    cleanup_confirmed: bool
    fresh_authoritative_state: bool
    no_stale_stream_flight: bool
    valid: bool
    collision: bool = False
    disqualified: bool = False
    correct_gate_sequence: bool = False
    completed: bool = False
    race_time_s: Optional[float] = None
    centering_margin: Optional[float] = None
    stability_margin: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bool_fields = (
            self.success,
            self.reset_epoch_proved,
            self.countdown_go_observed,
            self.watchdogs_armed,
            self.cleanup_confirmed,
            self.fresh_authoritative_state,
            self.no_stale_stream_flight,
            self.valid,
            self.collision,
            self.disqualified,
            self.correct_gate_sequence,
            self.completed,
        )
        if any(type(value) is not bool for value in bool_fields):
            raise TypeError("every powered safety/completion field must be an exact bool")
        for name in ("race_time_s", "centering_margin", "stability_margin"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in {int, float} or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric evidence")
        if self.race_time_s is not None and self.race_time_s < 0.0:
            raise ValueError("race_time_s must be non-negative")
        if not isinstance(self.details, Mapping):
            raise TypeError("powered trial details must be an object")
        json_hash(self.details)

    @property
    def safety_passed(self) -> bool:
        return all(
            (
                self.success,
                self.reset_epoch_proved,
                self.countdown_go_observed,
                self.watchdogs_armed,
                self.cleanup_confirmed,
                self.fresh_authoritative_state,
                self.no_stale_stream_flight,
                self.valid,
                not self.collision,
                not self.disqualified,
                self.correct_gate_sequence,
                self.completed,
            )
        )


class WarmCampaignBackend(Protocol):
    """Read-only declaration shape used to freeze a campaign plan."""

    simulator_build: str
    offline_during_timed_run: bool
    non_interactive_during_timed_run: bool
    maximum_powered_trial_s: float
    candidate_code_mode: str
    powered_watchdog_declaration: Mapping[str, Any]


def freeze_backend_contract(backend: WarmCampaignBackend) -> Mapping[str, Any]:
    """Validate and freeze a planning declaration, not an execution proof."""

    simulator_build = backend.simulator_build
    maximum = backend.maximum_powered_trial_s
    mode = backend.candidate_code_mode
    watchdog = backend.powered_watchdog_declaration
    if simulator_build != SIMULATOR_BUILD:
        raise ValueError(f"backend simulator_build must be exactly {SIMULATOR_BUILD}")
    if backend.offline_during_timed_run is not True or backend.non_interactive_during_timed_run is not True:
        raise ValueError("timed backend must be offline and non-interactive")
    if (
        type(maximum) not in {int, float}
        or not math.isfinite(maximum)
        or maximum <= 0.0
        or maximum > 60.0
    ):
        raise ValueError(
            "backend must declare a finite powered bound <=60s"
        )
    if type(mode) is not str or mode not in {
        "external-process-per-candidate",
        "verified-hot-swap",
    }:
        raise ValueError(
            "backend must declare external-process-per-candidate or verified-hot-swap"
        )
    expected_watchdog = {
        "schema",
        "mechanism",
        "maximum_powered_trial_s",
        "hard_stop_before_return",
        "implementation_sha256",
    }
    if (
        type(watchdog) is not dict
        or set(watchdog) != expected_watchdog
        or watchdog.get("schema") != "aigp-powered-watchdog-declaration/1"
        or type(watchdog.get("mechanism")) is not str
        or not watchdog["mechanism"].strip()
        or type(watchdog.get("maximum_powered_trial_s")) not in {int, float}
        or not math.isfinite(watchdog["maximum_powered_trial_s"])
        or float(watchdog["maximum_powered_trial_s"]) != float(maximum)
        or watchdog.get("hard_stop_before_return") is not True
        or type(watchdog.get("implementation_sha256")) is not str
        or len(watchdog["implementation_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in watchdog["implementation_sha256"]
        )
    ):
        raise ValueError(
            "backend must provide an exact watchdog planning declaration"
        )
    return {
        "schema": "aigp-warm-campaign-backend-contract/1",
        "simulator_build": simulator_build,
        "offline_during_timed_run": True,
        "non_interactive_during_timed_run": True,
        "maximum_powered_trial_s": float(maximum),
        "candidate_code_mode": mode,
        "powered_watchdog_declaration": dict(watchdog),
    }


def validate_campaign_definition(
    simulator_build: str,
    candidates: Sequence[CampaignCandidate],
    *,
    baseline_every: int,
    backend_contract: Mapping[str, Any],
) -> None:
    if simulator_build != SIMULATOR_BUILD:
        raise ValueError(f"simulator build must be exactly {SIMULATOR_BUILD}")
    if type(baseline_every) is not int or baseline_every < 1:
        raise ValueError("baseline_every must be a positive exact integer")
    if not candidates or any(type(item) is not CampaignCandidate for item in candidates):
        raise ValueError("campaign requires typed candidates")
    if sum(item.is_baseline for item in candidates) != 1:
        raise ValueError("campaign requires exactly one known-good baseline")
    if any(item.stage not in POWERED_STAGES for item in candidates):
        raise ValueError("powered campaign stage is outside the reviewed allowlist")
    if len({item.stage for item in candidates}) != 1:
        raise ValueError(
            "baseline drift evidence requires one shared powered campaign stage"
        )
    identifiers = [item.trial_id for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("each campaign execution needs a distinct trial row")
    if (
        type(backend_contract) is not dict
        or set(backend_contract)
        != {
            "schema",
            "simulator_build",
            "offline_during_timed_run",
            "non_interactive_during_timed_run",
            "maximum_powered_trial_s",
            "candidate_code_mode",
            "powered_watchdog_declaration",
        }
        or backend_contract.get("schema")
        != "aigp-warm-campaign-backend-contract/1"
        or backend_contract.get("simulator_build") != simulator_build
    ):
        raise ValueError("campaign backend contract/build is malformed or mismatched")
    maximum = backend_contract["maximum_powered_trial_s"]
    watchdog = backend_contract["powered_watchdog_declaration"]
    if (
        backend_contract["offline_during_timed_run"] is not True
        or backend_contract["non_interactive_during_timed_run"] is not True
        or type(maximum) not in {int, float}
        or not math.isfinite(maximum)
        or not 0.0 < maximum <= 60.0
        or type(backend_contract["candidate_code_mode"]) is not str
        or backend_contract["candidate_code_mode"]
        not in {"external-process-per-candidate", "verified-hot-swap"}
        or type(watchdog) is not dict
        or set(watchdog)
        != {
            "schema",
            "mechanism",
            "maximum_powered_trial_s",
            "hard_stop_before_return",
            "implementation_sha256",
        }
        or watchdog.get("schema") != "aigp-powered-watchdog-declaration/1"
        or type(watchdog.get("mechanism")) is not str
        or not watchdog["mechanism"].strip()
        or type(watchdog.get("maximum_powered_trial_s")) not in {int, float}
        or not math.isfinite(watchdog["maximum_powered_trial_s"])
        or float(watchdog["maximum_powered_trial_s"]) != float(maximum)
        or watchdog.get("hard_stop_before_return") is not True
        or type(watchdog.get("implementation_sha256")) is not str
        or len(watchdog["implementation_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in watchdog["implementation_sha256"]
        )
    ):
        raise ValueError("campaign backend watchdog planning declaration is malformed")
    # Canonicalization rejects non-finite or unserializable nested evidence.
    canonical_json(backend_contract)


def expanded_execution_schedule(
    candidates: Sequence[CampaignCandidate], *, baseline_every: int
) -> tuple[Mapping[str, Any], ...]:
    """Expand one baseline source into distinct periodic plan occurrences."""

    if type(baseline_every) is not int or baseline_every < 1:
        raise ValueError("baseline_every must be a positive exact integer")
    baselines = [item for item in candidates if item.is_baseline]
    if len(baselines) != 1:
        raise ValueError("campaign requires exactly one known-good baseline")
    baseline = baselines[0]
    pending = [item for item in candidates if not item.is_baseline]
    ordered: list[CampaignCandidate] = [baseline]
    for index, candidate in enumerate(pending, start=1):
        ordered.append(candidate)
        if index % baseline_every == 0:
            ordered.append(baseline)
    if pending and ordered[-1] is not baseline:
        ordered.append(baseline)
    counts: dict[str, int] = {}
    result = []
    for ordinal, candidate in enumerate(ordered):
        occurrence = counts.get(candidate.trial_id, 0)
        counts[candidate.trial_id] = occurrence + 1
        result.append(
            {
                "ordinal": ordinal,
                "occurrence_id": f"{ordinal:04d}:{candidate.trial_id}:{occurrence:03d}",
                "source_trial_id": candidate.trial_id,
                "label": candidate.label,
                "stage": candidate.stage,
                "is_baseline": candidate.is_baseline,
                "code_hash": candidate.code_hash,
                "config_hash": candidate.config_hash,
                "dataset_hash": candidate.dataset_hash,
                "evaluator_version": candidate.evaluator_version,
            }
        )
    return tuple(result)


def campaign_plan_hash(
    simulator_build: str,
    candidates: Sequence[CampaignCandidate],
    *,
    baseline_every: int,
    backend_contract: Mapping[str, Any],
) -> str:
    validate_campaign_definition(
        simulator_build,
        candidates,
        baseline_every=baseline_every,
        backend_contract=backend_contract,
    )
    schedule = expanded_execution_schedule(
        candidates, baseline_every=baseline_every
    )
    return json_hash(
        {
            "schema": "aigp-live-campaign-authorization/2",
            "simulator_build": simulator_build,
            "baseline_every": baseline_every,
            "backend_contract": dict(backend_contract),
            "execution_schedule": list(schedule),
        }
    )


def required_authorization_phrase(simulator_build: str, plan_hash: str) -> str:
    return f"AUTHORIZE_POWERED_VQ2:{simulator_build}:{plan_hash}"


class WarmCampaign:
    """Freeze a campaign plan; powered execution is intentionally unavailable.

    The authorization phrase is derived from the simulator build and exact
    ordered candidate plan, so authorization for an old/smaller campaign
    cannot silently authorize a changed one.
    """

    def __init__(
        self,
        ledger: TrialLedger,
        backend: WarmCampaignBackend,
        candidates: Sequence[CampaignCandidate],
        *,
        baseline_every: int = 5,
    ) -> None:
        backend_contract = freeze_backend_contract(backend)
        validate_campaign_definition(
            backend_contract["simulator_build"],
            candidates,
            baseline_every=baseline_every,
            backend_contract=backend_contract,
        )
        for candidate in candidates:
            row = ledger.get_trial(candidate.trial_id)
            expected = {
                "code_hash": candidate.code_hash,
                "config_hash": candidate.config_hash,
                "dataset_hash": candidate.dataset_hash,
                "evaluator_version": candidate.evaluator_version,
            }
            actual = {name: str(row[name]) for name in expected}
            if actual != expected:
                raise ValueError(
                    f"frozen campaign provenance mismatch for {candidate.trial_id}"
                )
            from .promotion import validate_promotion_chain

            validate_promotion_chain(ledger, candidate.trial_id)
        self.ledger = ledger
        self.candidates = tuple(candidates)
        self.baseline_every = baseline_every
        self.execution_schedule = expanded_execution_schedule(
            self.candidates, baseline_every=baseline_every
        )
        self.backend = backend
        self._backend_contract_json = canonical_json(backend_contract)
        self.simulator_build = str(backend_contract["simulator_build"])
        self.plan_hash = campaign_plan_hash(
            self.simulator_build,
            tuple(candidates),
            baseline_every=baseline_every,
            backend_contract=backend_contract,
        )

    @property
    def authorization_phrase(self) -> str:
        return required_authorization_phrase(self.simulator_build, self.plan_hash)

    def run(self, *, authorization: str) -> list[Mapping[str, Any]]:
        """Revalidate the frozen plan, then refuse before any powered action."""

        try:
            current_backend_contract = canonical_json(
                freeze_backend_contract(self.backend)
            )
        except Exception as exc:
            raise RuntimeError("powered backend contract is no longer valid") from exc
        if current_backend_contract != self._backend_contract_json:
            raise RuntimeError("powered backend contract changed after authorization")
        from .promotion import validate_promotion_chain

        for candidate in self.candidates:
            row = self.ledger.get_trial(candidate.trial_id)
            if any(
                row[name] != getattr(candidate, name)
                for name in (
                    "code_hash",
                    "config_hash",
                    "dataset_hash",
                    "evaluator_version",
                )
            ):
                raise RuntimeError("campaign source provenance changed after planning")
            validate_promotion_chain(self.ledger, candidate.trial_id)
        if authorization != self.authorization_phrase:
            raise PermissionError(
                "powered campaign not authorized for this exact build and plan"
            )
        # An authorization phrase and caller declaration are necessary plan
        # inputs, but neither proves a watchdog owns the powered process tree.
        # Refuse before acquiring a lease, creating a T5 child, invoking a
        # backend method, starting a heartbeat, or touching powered resources.
        raise RuntimeError(
            "powered campaign execution is unavailable: this repository does "
            "not ship a pinned out-of-process watchdog supervisor that owns "
            "hard-stop containment; the backend watchdog mapping is planning "
            "metadata only"
        )
