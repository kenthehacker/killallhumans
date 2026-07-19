"""Freeze a powered VQ2 campaign plan and derive its authorization phrase.

This command never starts FlightSim or a powered stage.  This repository ships
no powered executor; the watchdog mapping is planning metadata only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aigp_loop.campaign import (
    CampaignCandidate,
    campaign_plan_hash,
    expanded_execution_schedule,
    required_authorization_phrase,
    validate_campaign_definition,
)
from aigp_loop._util import json_hash, strict_json_load
from aigp_loop.ledger import TrialLedger
from aigp_loop.promotion import validate_promotion_chain


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="JSON object with a candidates array")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--simulator-build", required=True)
    parser.add_argument("--baseline-every", type=int, default=5)
    args = parser.parse_args(argv)
    payload = strict_json_load(args.plan)
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "backend_contract", "candidates"}
        or payload.get("schema") != "aigp-live-campaign-plan-input/1"
        or type(payload.get("candidates")) is not list
        or not payload["candidates"]
        or any(type(row) is not dict for row in payload["candidates"])
    ):
        parser.error("plan must have the exact versioned campaign input schema")
    try:
        candidates = tuple(CampaignCandidate(**row) for row in payload["candidates"])
        validate_campaign_definition(
            args.simulator_build,
            candidates,
            baseline_every=args.baseline_every,
            backend_contract=payload["backend_contract"],
        )
        ledger = TrialLedger(args.ledger)
        for candidate in candidates:
            row = ledger.get_trial(candidate.trial_id)
            for name in (
                "code_hash",
                "config_hash",
                "dataset_hash",
                "evaluator_version",
            ):
                if row[name] != getattr(candidate, name):
                    raise ValueError(
                        f"ledger provenance mismatch for {candidate.trial_id}"
                    )
            validate_promotion_chain(ledger, candidate.trial_id)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    digest = campaign_plan_hash(
        args.simulator_build,
        candidates,
        baseline_every=args.baseline_every,
        backend_contract=payload["backend_contract"],
    )
    schedule = expanded_execution_schedule(
        candidates, baseline_every=args.baseline_every
    )
    print(
        json.dumps(
            {
                "schema": "aigp-live-campaign-plan/1",
                "simulator_build": args.simulator_build,
                "baseline_every": args.baseline_every,
                "backend_contract_sha256": json_hash(payload["backend_contract"]),
                "plan_hash": digest,
                "execution_schedule": list(schedule),
                "authorization_phrase": required_authorization_phrase(
                    args.simulator_build, digest
                ),
                "execution_started": False,
                "notice": (
                    "Plan only. This repository has no powered executor or "
                    "pinned watchdog supervisor; the phrase does not enable power."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
