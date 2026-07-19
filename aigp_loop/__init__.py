"""Durable, replay-first development loop for the AIGP autonomy stack.

The package deliberately contains no simulator import or implicit live entry
point.  Non-live promotion, replay, and trial-ledger tooling is safe to use in
CI; powered campaigns require a separate explicit authorization boundary.
"""

from .ledger import TrialKey, TrialLedger
from .promotion import PromotionDecision, PromotionLadder, Tier

__all__ = [
    "PromotionDecision",
    "PromotionLadder",
    "Tier",
    "TrialKey",
    "TrialLedger",
]
