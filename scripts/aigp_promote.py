"""Apply the safety-first successive-halving decision to evaluation JSON."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aigp_loop.promotion import (
    CandidateEvaluation,
    HardGates,
    PromotionLadder,
    QualityVector,
    Tier,
    TierEligibility,
)
from aigp_loop._util import strict_json_load


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", help="JSON array of candidate evaluation objects")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-survivors", type=int, default=1)
    args = parser.parse_args(argv)
    payload = strict_json_load(args.evaluations)
    if type(payload) is not list or not payload:
        raise ValueError("evaluations must be a non-empty JSON array")
    evaluations = [_parse_evaluation(row) for row in payload]
    decision = PromotionLadder(
        keep_fraction=args.keep_fraction,
        minimum_survivors=args.minimum_survivors,
    ).decide(evaluations)
    result = dataclasses.asdict(decision)
    result["tier"] = int(decision.tier)
    result["next_tier"] = int(decision.next_tier) if decision.next_tier is not None else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_evaluation(row: Any) -> CandidateEvaluation:
    required = {"candidate_id", "tier"}
    allowed = required | {"hard_gates", "eligibility", "quality", "repetitions", "metrics"}
    if type(row) is not dict or not required <= set(row) or set(row) - allowed:
        raise ValueError("evaluation has missing or unknown fields")
    if type(row["candidate_id"]) is not str:
        raise TypeError("candidate_id must be an exact string")
    if type(row["tier"]) is not int:
        raise TypeError("tier must be an exact integer")
    quality: Any = row.get("quality", {})
    quality_fields = {
        "completion_reliability",
        "centering_margin",
        "stability_margin",
        "race_time_s",
    }
    if type(quality) is not dict or set(quality) - quality_fields:
        raise ValueError("quality must contain only recognized fields")
    repetitions = row.get("repetitions", 1)
    if type(repetitions) is not int:
        raise TypeError("repetitions must be an exact integer")
    metrics: Mapping[str, Any] = row.get("metrics", {})
    if type(metrics) is not dict:
        raise TypeError("metrics must be an exact object")
    tier = Tier(row["tier"])
    hard_gates = None
    eligibility = None
    if tier <= Tier.T1_VQ2_REPLAY:
        if "hard_gates" in row or type(row.get("eligibility")) is not dict:
            raise ValueError("T0/T1 require eligibility and forbid hard_gates")
        raw_eligibility = row["eligibility"]
        if set(raw_eligibility) - {"scope", "passed", "evidence_hash", "failures"} or not {
            "scope", "passed"
        } <= set(raw_eligibility):
            raise ValueError("eligibility has missing or unknown fields")
        failures = raw_eligibility.get("failures", [])
        if type(failures) is not list:
            raise TypeError("eligibility failures must be an array")
        eligibility = TierEligibility(
            scope=raw_eligibility["scope"],
            passed=raw_eligibility["passed"],
            evidence_hash=raw_eligibility.get("evidence_hash"),
            failures=tuple(failures),
        )
    else:
        if "eligibility" in row or "hard_gates" not in row:
            raise ValueError("T2+ require hard_gates and forbid eligibility")
        hard_gates = HardGates.from_mapping(row["hard_gates"])
    return CandidateEvaluation(
        candidate_id=row["candidate_id"],
        tier=tier,
        hard_gates=hard_gates,
        quality=QualityVector(**quality),
        repetitions=repetitions,
        metrics=metrics,
        eligibility=eligibility,
    )


if __name__ == "__main__":
    raise SystemExit(main())
