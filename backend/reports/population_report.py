"""
CardioAI Pro — Payer Population Report
==========================================
Aggregates every claims-pipeline record the engine has processed into a
population-level view for a payer: risk tier breakdown, a high-risk
cohort list, and the qualitative narrative (MLR impact, Star Rating
uplift) a payer-facing report would carry.

The per-member risk scores this rolls up come from the same placeholder
heuristic as everywhere else in this engine — see inference/models.py —
so this report inherits that "not clinically validated" caveat at the
population level. It deliberately does NOT compute a dollar-figure medical
cost avoidance estimate or a specific CMS Star Rating point impact: both
require real claims/cost data and real measure-level mapping this engine
doesn't have, and inventing plausible-looking numbers for them would be
worse than not reporting them at all.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemberRiskRecord:
    member_id: str
    age: int
    risk_flags_count: int
    score: float
    risk_tier: str
    recorded_at: float
    claim_history: list[dict[str, Any]] = None  # [{claim_id, charge_amount, diagnosis_codes, recorded_at}] — real cost data, when available (via the X12 837 parser)

    def __post_init__(self):
        if self.claim_history is None:
            self.claim_history = []


class PopulationAggregator:
    def __init__(self, max_members: int = 5000):
        self.members: dict[str, MemberRiskRecord] = {}
        self.max_members = max_members

    def record(
        self, member_id: str, age: int, risk_flags_count: int, score: float, risk_tier: str,
        claim_id: str | None = None, total_charge_amount: float | None = None, diagnosis_codes: list[str] | None = None,
    ) -> None:
        existing = self.members.get(member_id)
        claim_history = existing.claim_history if existing else []
        if total_charge_amount is not None:
            claim_history = claim_history + [{
                "claim_id": claim_id, "charge_amount": total_charge_amount,
                "diagnosis_codes": diagnosis_codes or [], "recorded_at": time.time(),
            }]
        self.members[member_id] = MemberRiskRecord(
            member_id, age, risk_flags_count, score, risk_tier, time.time(), claim_history,
        )
        if len(self.members) > self.max_members:
            oldest = min(self.members.values(), key=lambda m: m.recorded_at)
            del self.members[oldest.member_id]

    def get_member(self, member_id: str) -> Optional[dict[str, Any]]:
        m = self.members.get(member_id)
        if not m:
            return None
        return {
            "member_id": m.member_id, "age": m.age, "risk_flags_count": m.risk_flags_count,
            "score": m.score, "risk_tier": m.risk_tier, "recorded_at": m.recorded_at,
            "claim_history": m.claim_history,
        }

    def report(self, top_n: int = 20) -> dict[str, Any]:
        members = list(self.members.values())
        n = len(members)
        tier_counts = {"low": 0, "moderate": 0, "high": 0}
        for m in members:
            tier_counts[m.risk_tier] = tier_counts.get(m.risk_tier, 0) + 1
        avg_score = round(sum(m.score for m in members) / n, 3) if n else 0.0
        high_risk_cohort = sorted(
            [m for m in members if m.risk_tier == "high"], key=lambda m: m.score, reverse=True
        )[:top_n]
        high_pct = round(100 * tier_counts["high"] / n, 1) if n else 0.0

        return {
            "generated_at": time.time(),
            "total_members": n,
            "tier_breakdown": tier_counts,
            "avg_risk_score": avg_score,
            "high_risk_cohort": [
                {"member_id": m.member_id, "age": m.age, "risk_flags_count": m.risk_flags_count, "score": m.score}
                for m in high_risk_cohort
            ],
            "narrative": {
                "mlr_impact": (
                    f"{tier_counts['high']} of {n} members flagged high-risk ({high_pct}%). "
                    "Early intervention in this cohort is the mechanism a medical-cost-avoidance claim would "
                    "rest on — no dollar figure is computed here without real claims/cost data to validate against."
                ) if n else "No claims records processed yet.",
                "star_rating_note": (
                    "CMS Star Rating impact depends on which specific measures (e.g. medication adherence, "
                    "readmission rates) this cohort maps to — not something derivable from risk scores alone."
                ),
            },
        }
