"""Tier-scoped evidence contracts without live-evidence relabeling.

This module describes what each promotion tier is allowed to claim.  It does
not replace the established replay/non-live payload schemas, and T5 has no
accepted contract because this repository ships no powered executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Optional

from .promotion import Tier


_REPLAY_SCHEMAS = (
    "aigp-vq2-replay-score/1",
    "aigp-vq2-replay-corpus-score/1",
)
_NONLIVE_SCHEMA = "aigp-nonlive-promotion-evidence/1"

# T0 and T1 are deliberately non-flight evidence domains.  Merely spelling a
# flight-domain assertion with ``False`` still changes the meaning of the
# payload (for example, from "no flight claim" to "we verified no powered
# resources were used").  Keep those claims out of the payload entirely; the
# tier scope above is the sole authority for their meaning.
_CLAIM_VALUE_KEYS = frozenset(
    {"domain", "execution", "executionmode", "mode", "schema", "simtype", "simulator"}
)
_CLAIM_VALUE_WORDS = frozenset(
    {
        "assertion",
        "attestation",
        "claim",
        "claims",
        "domain",
        "execution",
        "mode",
        "schema",
        "sim",
        "simulation",
        "simulator",
        "verification",
        "verified",
    }
)
_DIRECT_CLAIM_FRAGMENTS = (
    "closedloop",
    "gateauthority",
    "officialflightsim",
    "officialsimulator",
    "powered",
)
_FORBIDDEN_VALUE_FRAGMENTS = (
    "closedloop",
    "gateauthority",
    "officialflightsim",
    "officialsimulator",
    "powered",
)
_NONFLIGHT_RESERVED_KEY_WORDS = frozenset({"armed", "flight", "live"})
_ALL_NONLIVE_RESERVED_KEY_WORDS = frozenset({"armed", "live"})
_NONFLIGHT_RESERVED_KEY_COMPOUNDS = (
    frozenset({"gate", "passed"}),
)
_SCALAR_CLAIM_VALUE_WORDS = frozenset(
    {
        "authorization",
        "backend",
        "context",
        "environment",
        "outcome",
        "phase",
        "resource",
        "resources",
        "result",
        "scope",
        "state",
        "status",
        "type",
        "verdict",
    }
)
_RESERVED_CLAIM_VALUE_WORDS = frozenset({"armed", "flight", "live", "official"})
_RESERVED_CLAIM_VALUE_COMPOUNDS = (
    frozenset({"closed", "loop"}),
    frozenset({"gate", "authority"}),
    frozenset({"gate", "passed"}),
)
_T2_TO_T4_REQUIRED_PROVENANCE = {
    "execution": "deterministic_synthetic_kinematic_nonpowered",
    "powered_resources_used": False,
    "cleanup_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
    "stale_stream_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
    "centering_proxy": "negative_worst_p95_tracking_error_m",
    "stability_proxy": "negative_worst_max_tracking_error_m",
}
_T1_REQUIRED_PROVENANCE = {
    "candidate": {
        "perception": "candidate_detector_on_all_decoded_frames",
        "estimator": "candidate_estimator_on_ordered_sanitized_stream",
        "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
    },
    "recorded": {
        "perception": "recorded_processed_frames",
        "estimator": "recorded_bundle_context",
        "open_loop_commands": "recorded_bundle_command_stream",
    },
}


class EvidenceDomain(str, Enum):
    AFFECTED_TESTS = "affected_tests"
    REPLAY_OPEN_LOOP = "replay_open_loop"
    SYNTHETIC_CLOSED_LOOP = "synthetic_closed_loop"


class GateAuthorityClaim(str, Enum):
    NONE = "none"
    SYNTHETIC_SEQUENCE = "synthetic_sequence"


@dataclass(frozen=True, slots=True)
class TierEvidenceScopeV1:
    """Exact claim boundary for one ordinary T0--T4 tier."""

    SCHEMA: ClassVar[str] = "aigp-tier-evidence-scope/1"

    tier: Tier
    domain: EvidenceDomain
    closed_loop: bool
    powered: bool
    gate_authority: GateAuthorityClaim
    payload_schemas: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.tier) is not Tier or not Tier.T0_AFFECTED <= self.tier <= Tier.T4_FULL_NON_LIVE:
            raise ValueError("tier evidence scope supports only ordinary T0-T4")
        if type(self.domain) is not EvidenceDomain:
            raise TypeError("domain must be EvidenceDomain")
        if type(self.closed_loop) is not bool or type(self.powered) is not bool:
            raise TypeError("closed_loop and powered must be exact bools")
        if type(self.gate_authority) is not GateAuthorityClaim:
            raise TypeError("gate_authority must be GateAuthorityClaim")
        if type(self.payload_schemas) is not tuple or any(
            type(item) is not str or not item for item in self.payload_schemas
        ):
            raise TypeError("payload_schemas must be a tuple of non-empty strings")
        expected = scope_for_tier(self.tier, _constructing=True)
        observed = (
            self.domain,
            self.closed_loop,
            self.powered,
            self.gate_authority,
            self.payload_schemas,
        )
        if expected != observed:
            raise ValueError("tier evidence scope attempts to relabel its evidence domain")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tier": int(self.tier),
            "domain": self.domain.value,
            "closed_loop": self.closed_loop,
            "powered": self.powered,
            "gate_authority": self.gate_authority.value,
            "payload_schemas": list(self.payload_schemas),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "TierEvidenceScopeV1":
        keys = {
            "schema",
            "tier",
            "domain",
            "closed_loop",
            "powered",
            "gate_authority",
            "payload_schemas",
        }
        if type(value) is not dict or set(value) != keys:
            raise ValueError("tier evidence scope fields must be exact")
        if value["schema"] != cls.SCHEMA:
            raise ValueError("unsupported tier evidence scope schema")
        if type(value["tier"]) is not int:
            raise TypeError("tier must be an exact integer")
        if type(value["domain"]) is not str or type(value["gate_authority"]) is not str:
            raise TypeError("evidence domain/authority must be exact strings")
        if type(value["closed_loop"]) is not bool or type(value["powered"]) is not bool:
            raise TypeError("closed_loop and powered must be exact bools")
        if type(value["payload_schemas"]) is not list:
            raise TypeError("payload_schemas must be an array")
        try:
            tier = Tier(value["tier"])
            domain = EvidenceDomain(value["domain"])
            authority = GateAuthorityClaim(value["gate_authority"])
        except ValueError as exc:
            raise ValueError("tier evidence scope contains an unknown enum value") from exc
        return cls(
            tier=tier,
            domain=domain,
            closed_loop=value["closed_loop"],
            powered=value["powered"],
            gate_authority=authority,
            payload_schemas=tuple(value["payload_schemas"]),
        )


def scope_for_tier(
    tier: Tier,
    *,
    _constructing: bool = False,
) -> TierEvidenceScopeV1 | tuple[
    EvidenceDomain,
    bool,
    bool,
    GateAuthorityClaim,
    tuple[str, ...],
]:
    """Return the fixed claim boundary; T5 is intentionally unavailable."""

    if type(tier) is not Tier:
        raise TypeError("tier must be a Tier value")
    if tier is Tier.T0_AFFECTED:
        values = (
            EvidenceDomain.AFFECTED_TESTS,
            False,
            False,
            GateAuthorityClaim.NONE,
            (),
        )
    elif tier is Tier.T1_VQ2_REPLAY:
        values = (
            EvidenceDomain.REPLAY_OPEN_LOOP,
            False,
            False,
            GateAuthorityClaim.NONE,
            _REPLAY_SCHEMAS,
        )
    elif Tier.T2_WARM_SIM <= tier <= Tier.T4_FULL_NON_LIVE:
        values = (
            EvidenceDomain.SYNTHETIC_CLOSED_LOOP,
            True,
            False,
            GateAuthorityClaim.SYNTHETIC_SEQUENCE,
            (_NONLIVE_SCHEMA,),
        )
    else:
        raise ValueError("T5 evidence is unavailable without a reviewed powered executor")
    if _constructing:
        return values
    return TierEvidenceScopeV1(
        tier=tier,
        domain=values[0],
        closed_loop=values[1],
        powered=values[2],
        gate_authority=values[3],
        payload_schemas=values[4],
    )


def find_unique_schema_evidence(
    metrics: Mapping[str, Any],
    schemas: set[str] | frozenset[str],
) -> Optional[Mapping[str, Any]]:
    """Find exactly one matching payload, rejecting ambiguous nesting."""

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if type(schemas) not in {set, frozenset} or not schemas or any(
        type(item) is not str or not item for item in schemas
    ):
        raise TypeError("schemas must be a non-empty exact set of strings")
    # The boolean marks one direct child of a canonical corpus ``sessions``
    # array.  That child's replay-score root belongs to the corpus envelope and
    # is therefore not a second evidence unit.  Its descendants still need to
    # be searched so an extra replay payload cannot hide inside the session.
    pending: list[tuple[Any, bool]] = [(metrics, False)]
    matches: list[Mapping[str, Any]] = []
    visited: set[int] = set()
    while pending:
        value, suppress_session_root = pending.pop()
        if isinstance(value, Mapping):
            schema = value.get("schema")
            if schema in schemas and not (
                suppress_session_root and schema == "aigp-vq2-replay-score/1"
            ):
                matches.append(value)
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            matching_corpus = (
                schema == "aigp-vq2-replay-corpus-score/1" and schema in schemas
            )
            pending.extend(
                (child, matching_corpus and key == "sessions")
                for key, child in value.items()
            )
        elif type(value) is list:
            pending.extend((child, suppress_session_root) for child in value)
    if len(matches) > 1:
        raise ValueError("metrics contain ambiguous duplicate tier evidence")
    return matches[0] if matches else None


def _claim_token(value: str) -> str:
    """Normalize snake/kebab/camel/punctuation variants to one token."""

    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _claim_words(value: str) -> tuple[str, ...]:
    """Split snake/kebab/camel/acronym spellings without fuzzy matching."""

    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.strip())
    camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split)
    return tuple(part.lower() for part in re.findall(r"[A-Za-z0-9]+", camel_split))


def _is_t0_test_identifier_key(value: str) -> bool:
    """Recognize bounded pytest identifiers, not arbitrary claim-key aliases."""

    if re.fullmatch(r"test_[A-Za-z0-9_]+(?:\[[^\]\r\n]{1,200}\])?", value):
        return True
    normalized = value.replace("\\", "/")
    path = normalized.split("::", 1)[0]
    segments = path.split("/")
    return (
        "tests" in segments[:-1]
        and re.fullmatch(r"test_[A-Za-z0-9_.-]+\.py", segments[-1]) is not None
    )


def _is_direct_flight_claim_key(
    tier: Tier, key: str, token: str, words: tuple[str, ...]
) -> bool:
    if tier is Tier.T0_AFFECTED and _is_t0_test_identifier_key(key):
        return False
    word_set = frozenset(words)
    return (
        any(fragment in token for fragment in _DIRECT_CLAIM_FRAGMENTS)
        or {"closed", "loop"} <= word_set
        or {"gate", "authority"} <= word_set
        or {"live", "flight"} <= word_set
        or "flight" in word_set
        and bool({"sim", "simulation", "simulator"} & word_set)
        or "official" in word_set
        and bool({"sim", "simulation", "simulator"} & word_set)
        or bool(_ALL_NONLIVE_RESERVED_KEY_WORDS & word_set)
        or tier <= Tier.T1_VQ2_REPLAY
        and (
            bool(_NONFLIGHT_RESERVED_KEY_WORDS & word_set)
            or any(
                reserved <= word_set
                for reserved in _NONFLIGHT_RESERVED_KEY_COMPOUNDS
            )
        )
    )


def _is_claim_value_key(token: str, words: tuple[str, ...]) -> bool:
    return token in _CLAIM_VALUE_KEYS or bool(_CLAIM_VALUE_WORDS & set(words))


def _is_scalar_claim_value_key(words: tuple[str, ...]) -> bool:
    return bool(_SCALAR_CLAIM_VALUE_WORDS & set(words))


def _is_forbidden_claim_value(
    tier: Tier, value: str, *, schema: bool
) -> bool:
    token = _claim_token(value)
    words = frozenset(_claim_words(value))
    if any(fragment in token for fragment in _FORBIDDEN_VALUE_FRAGMENTS):
        return True
    reserved_words = _RESERVED_CLAIM_VALUE_WORDS & words
    if tier >= Tier.T2_WARM_SIM and {"non", "live"} <= words:
        reserved_words -= {"live"}
    if reserved_words or any(
        reserved <= words for reserved in _RESERVED_CLAIM_VALUE_COMPOUNDS
    ):
        return True
    if "flight" in words and bool({"sim", "simulation", "simulator"} & words):
        return True
    schema_token = (
        token.replace("nonlive", "")
        if tier >= Tier.T2_WARM_SIM
        else token
    )
    return schema and (
        ("live" in schema_token and "evidence" in schema_token)
        or ("flight" in schema_token and "evidence" in schema_token)
    )


def _contains_forbidden_claim_value(
    tier: Tier, value: Any, *, schema: bool
) -> bool:
    """Inspect strings in one structured claim value, including arrays."""

    pending = [value]
    visited: set[int] = set()
    while pending:
        child = pending.pop()
        if type(child) is str:
            if _is_forbidden_claim_value(tier, child, schema=schema):
                return True
        elif isinstance(child, Mapping) or type(child) in {list, tuple}:
            identity = id(child)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(child.values() if isinstance(child, Mapping) else child)
    return False


def _reject_scope_relabeling_claims(
    tier: Tier,
    metrics: Mapping[str, Any],
    *,
    allowed_synthetic_provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Reject recursively nested or spelling-obfuscated domain relabeling."""

    pending: list[tuple[str, Any]] = [("$", metrics)]
    visited: set[int] = set()
    while pending:
        path, value = pending.pop()
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if value is allowed_synthetic_provenance:
                # validate_tier_evidence checks this exact two-field mapping
                # before calling here. It is the one payload-owned scope claim.
                continue
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if type(key) is str:
                    token = _claim_token(key)
                    words = _claim_words(key)
                    if _is_direct_flight_claim_key(tier, key, token, words):
                        raise ValueError(
                            f"{tier.name} cannot contain flight-domain claim {child_path}"
                        )
                    schema_claim = token == "schema" or "schema" in words
                    forbidden_claim_value = (
                        child is not allowed_synthetic_provenance
                        and (
                            _is_claim_value_key(token, words)
                            and _contains_forbidden_claim_value(
                                tier, child, schema=schema_claim
                            )
                            or _is_scalar_claim_value_key(words)
                            and type(child) is str
                            and _is_forbidden_claim_value(
                                tier, child, schema=schema_claim
                            )
                        )
                    )
                    if forbidden_claim_value:
                        # The canonical T2-T4 non-live envelope is allowed;
                        # other live/powered schemas remain contradictory.
                        if not (
                            tier >= Tier.T2_WARM_SIM
                            and token == "schema"
                            and type(child) is str
                            and child == _NONLIVE_SCHEMA
                        ):
                            raise ValueError(
                                f"{tier.name} cannot contain flight-domain claim "
                                f"{child_path}={child!r}"
                            )
                pending.append((child_path, child))
        elif type(value) in {list, tuple}:
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(
                (f"{path}[{index}]", child)
                for index, child in enumerate(value)
            )


def validate_tier_evidence(
    tier: Tier,
    metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate schema/domain/track scope without claiming promotion success."""

    scope = scope_for_tier(tier)
    if not isinstance(metrics, Mapping):
        raise TypeError(f"{tier.name} metrics must be a mapping")
    if tier <= Tier.T1_VQ2_REPLAY:
        _reject_scope_relabeling_claims(tier, metrics)
    if tier is Tier.T0_AFFECTED:
        return metrics
    evidence = find_unique_schema_evidence(metrics, set(scope.payload_schemas))
    if evidence is None:
        raise ValueError(f"{tier.name} is missing its tier-scoped evidence")
    if tier is Tier.T1_VQ2_REPLAY:
        # Replay is causal/open-loop evidence even when it scores generated
        # commands.  Flight-domain assertions are absent rather than inferred
        # from a caller-provided boolean.
        provenance = evidence.get("domain_provenance")
        if type(provenance) is not dict:
            raise ValueError("T1 replay evidence requires exact domain provenance")
        base_keys = {"perception", "estimator", "open_loop_commands"}
        allowed_keys = base_keys | {"worker_transport"}
        if (
            not base_keys <= set(provenance)
            or not set(provenance) <= allowed_keys
            or not any(
                all(provenance.get(key) == value for key, value in expected.items())
                for expected in _T1_REQUIRED_PROVENANCE.values()
            )
            or (
                "worker_transport" in provenance
                and (
                    provenance["worker_transport"]
                    != "candidate_worktree_code_hash"
                    or any(
                        provenance.get(key) != value
                        for key, value in _T1_REQUIRED_PROVENANCE["candidate"].items()
                    )
                )
            )
        ):
            raise ValueError("T1 replay evidence requires exact causal open-loop provenance")
        return evidence

    from .nonlive import DOMAIN_TRACK_SET, FULL_TRACK_SET

    expected_tracks = {
        Tier.T2_WARM_SIM: ("race_01",),
        Tier.T3_DOMAIN_TRACKS: DOMAIN_TRACK_SET,
        Tier.T4_FULL_NON_LIVE: FULL_TRACK_SET,
    }[tier]
    observed_tracks = evidence.get("track_identity")
    provenance = evidence.get("domain_provenance")
    if (
        evidence.get("tier") != int(tier)
        or type(observed_tracks) is not list
        or any(type(item) is not str for item in observed_tracks)
        or tuple(sorted(observed_tracks)) != tuple(sorted(expected_tracks))
        or type(provenance) is not dict
        or provenance != _T2_TO_T4_REQUIRED_PROVENANCE
    ):
        raise ValueError(f"{tier.name} evidence has the wrong synthetic domain scope")
    _reject_scope_relabeling_claims(
        tier,
        metrics,
        allowed_synthetic_provenance=provenance,
    )
    return evidence


__all__ = [
    "EvidenceDomain",
    "GateAuthorityClaim",
    "TierEvidenceScopeV1",
    "find_unique_schema_evidence",
    "scope_for_tier",
    "validate_tier_evidence",
]
