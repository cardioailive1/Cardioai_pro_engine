"""
CardioAI Pro — Cost / Outcomes Linkage
============================================
Closes the gap flagged earlier: the population report deliberately never
computed a dollar-figure medical cost avoidance estimate, because it had
no real claims cost data to validate one against. Now that
integrations/x12_837.py can parse real claim charge amounts, this module
defines the actual methodology for turning a member's claims cost
HISTORY into an honest cost-avoidance estimate — and, just as
importantly, when NOT to report one.

THE METHODOLOGY, AND WHY IT'S STRUCTURED THIS WAY:
A cost-avoidance claim ("early intervention in this cohort saved $X") is
only meaningful as a COMPARISON — a flagged member's cost trajectory
against what it would plausibly have been without intervention. This
module uses the simplest defensible version of that comparison: split
each member's claim history at the point they were FIRST flagged
high-risk, fit a linear cost trend on the PRE-flag claims and a separate
trend on the POST-flag claims (reusing the same least-squares approach
longitudinal/trend_engine.py already uses for vitals, applied here to
dollars over time instead), and compare the two slopes. A flattening or
declining post-flag trend against a rising pre-flag trend is the
pattern a genuine early-intervention effect would produce.

THIS IS NOT A CONTROLLED COMPARISON, STATED PLAINLY: without a real
control group (similar members who were NOT flagged, or a
difference-in-differences design against a comparable population), a
pre/post trend change could also reflect regression to the mean, natural
care-seeking variation, or the specific illness episode the claims
happened to capture — not necessarily the platform's effect. This module
reports the pre/post trend comparison as what it is (a within-member
trend change) and does not claim it proves causation. A real production
deployment wanting a defensible cost-avoidance number for payer
reporting should extend this with an actual control cohort.

DATA-GATED, LIKE EVERYTHING ELSE COST-RELATED IN THIS PROJECT: a member
needs a minimum number of claims both before and after their flag date
before this module will compute anything for them — see MIN_CLAIMS_PER_PERIOD
below. Below that, it's noise, not a trend, and the report says so rather
than presenting a number computed from too little data as if it means
something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

MIN_CLAIMS_PER_PERIOD = 3  # need at least this many claims before AND after the flag date to fit a trend on either side
MIN_MEMBERS_FOR_POPULATION_ESTIMATE = 10  # population-level rollup needs enough members with individually-valid trends to not be dominated by one or two outliers


def _linear_slope(x: list[float], y: list[float]) -> Optional[float]:
    """Ordinary least squares slope — dollars per day. None if there's no meaningful time spread to fit against."""
    n = len(x)
    if n < 2:
        return None
    x_mean, y_mean = sum(x) / n, sum(y) / n
    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0:
        return None
    return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denom


@dataclass
class MemberCostTrend:
    member_id: str
    n_pre_claims: int
    n_post_claims: int
    sufficient_data: bool
    pre_flag_slope_per_day: Optional[float] = None   # dollars/day trend BEFORE the member was first flagged high-risk
    post_flag_slope_per_day: Optional[float] = None  # dollars/day trend AFTER
    trend_change: Optional[float] = None              # post - pre; negative means costs decelerated after flagging
    anchor_source: str = "flagged_at"  # "flagged_at" (fallback) | "intervention" (a real care task reached a terminal status) — see compute_member_cost_trend's docstring
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id, "n_pre_claims": self.n_pre_claims, "n_post_claims": self.n_post_claims,
            "sufficient_data": self.sufficient_data, "anchor_source": self.anchor_source,
            "pre_flag_slope_per_day": round(self.pre_flag_slope_per_day, 2) if self.pre_flag_slope_per_day is not None else None,
            "post_flag_slope_per_day": round(self.post_flag_slope_per_day, 2) if self.post_flag_slope_per_day is not None else None,
            "trend_change": round(self.trend_change, 2) if self.trend_change is not None else None,
            "narrative": self.narrative,
        }


def compute_member_cost_trend(
    member_id: str, claim_history: list[dict[str, Any]], flagged_at: float,
    intervention_started_at: Optional[float] = None,
) -> MemberCostTrend:
    """
    claim_history: [{charge_amount, recorded_at}, ...] — the same shape
    reports/population_report.py's MemberRiskRecord.claim_history carries.
    flagged_at: the timestamp this member was first scored high-risk —
    the fallback anchor.
    intervention_started_at: when available (from
    care_management.tasks.CareTaskRegistry.get_intervention_anchor() — the
    earliest terminal-status care task for this member), this is used
    INSTEAD of flagged_at. A risk score crossing a threshold and a care
    coordinator actually enrolling/referring/reaching the member are
    different events; anchoring on the real one, when it exists, is a
    materially better pre/post split than anchoring on the score alone.
    """
    anchor = intervention_started_at if intervention_started_at is not None else flagged_at
    anchor_source = "intervention" if intervention_started_at is not None else "flagged_at"
    pre = [c for c in claim_history if c["recorded_at"] < anchor and c.get("charge_amount") is not None]
    post = [c for c in claim_history if c["recorded_at"] >= anchor and c.get("charge_amount") is not None]

    sufficient = len(pre) >= MIN_CLAIMS_PER_PERIOD and len(post) >= MIN_CLAIMS_PER_PERIOD
    if not sufficient:
        return MemberCostTrend(
            member_id=member_id, n_pre_claims=len(pre), n_post_claims=len(post), sufficient_data=False,
            anchor_source=anchor_source,
            narrative=f"Only {len(pre)} pre-{anchor_source} and {len(post)} post-{anchor_source} claims — need at least {MIN_CLAIMS_PER_PERIOD} of each to fit a real trend, not noise.",
        )

    t0 = claim_history[0]["recorded_at"]
    DAY = 86400.0
    # x must be in DAYS, not raw seconds — recorded_at is a Unix timestamp,
    # and fitting the slope against raw seconds silently produces a
    # dollars-per-SECOND value while every label here says "per day": a
    # real ~$10/day trend would compute as ~0.000116/second, which rounds
    # to 0.00 at the 2-decimal display precision — a real bug caught by
    # actually running this against a constructed rising-cost trend and
    # noticing the reported slope was 0.00 when it obviously shouldn't be.
    pre_slope = _linear_slope([(c["recorded_at"] - t0) / DAY for c in pre], [c["charge_amount"] for c in pre])
    post_slope = _linear_slope([(c["recorded_at"] - t0) / DAY for c in post], [c["charge_amount"] for c in post])

    if pre_slope is None or post_slope is None:
        return MemberCostTrend(
            member_id=member_id, n_pre_claims=len(pre), n_post_claims=len(post), sufficient_data=False,
            anchor_source=anchor_source,
            narrative="Enough claims, but not enough time spread within one period to fit a reliable slope.",
        )

    change = post_slope - pre_slope
    direction = "decelerated" if change < 0 else "accelerated" if change > 0 else "stayed flat"
    anchor_desc = "after a real recorded intervention (enrollment/referral)" if anchor_source == "intervention" else "after flagging (no recorded intervention yet — using the risk-score date as a fallback anchor)"
    narrative = (
        f"Cost trend {direction} {anchor_desc}: {pre_slope:+.2f}/day before -> {post_slope:+.2f}/day after. "
        "This is a within-member trend comparison, not a controlled estimate — see module docstring."
    )
    return MemberCostTrend(
        member_id=member_id, n_pre_claims=len(pre), n_post_claims=len(post), sufficient_data=True,
        pre_flag_slope_per_day=pre_slope, post_flag_slope_per_day=post_slope, trend_change=change,
        anchor_source=anchor_source, narrative=narrative,
    )


def population_cost_avoidance_report(members_with_history: dict[str, tuple[list[dict[str, Any]], float, Optional[float]]]) -> dict[str, Any]:
    """
    members_with_history: member_id -> (claim_history, flagged_at,
    intervention_started_at). The third element may be None — passed
    straight through to compute_member_cost_trend, which falls back to
    flagged_at when it is. Returns an honest population-level rollup — a
    real number ONLY when enough members individually have sufficient
    data; otherwise says so instead of computing one from too few members
    and presenting it with false confidence.
    """
    trends = [
        compute_member_cost_trend(mid, hist, flagged_at, intervention_started_at=intervention_at)
        for mid, (hist, flagged_at, intervention_at) in members_with_history.items()
    ]
    valid = [t for t in trends if t.sufficient_data]

    if len(valid) < MIN_MEMBERS_FOR_POPULATION_ESTIMATE:
        return {
            "members_evaluated": len(trends), "members_with_sufficient_data": len(valid),
            "population_estimate_available": False,
            "narrative": (
                f"Only {len(valid)} of {len(trends)} members have enough pre/post claims history for a trend "
                f"comparison — need at least {MIN_MEMBERS_FOR_POPULATION_ESTIMATE} before a population-level "
                "estimate means anything more than a few individual trend lines."
            ),
            "member_trends": [t.to_dict() for t in trends],
        }

    avg_change = sum(t.trend_change for t in valid) / len(valid)
    improved = sum(1 for t in valid if t.trend_change < 0)
    return {
        "members_evaluated": len(trends), "members_with_sufficient_data": len(valid),
        "population_estimate_available": True,
        "avg_trend_change_per_day": round(avg_change, 2),
        "members_with_decelerating_cost": improved,
        "members_with_decelerating_cost_pct": round(100 * improved / len(valid), 1),
        "narrative": (
            f"Across {len(valid)} members with sufficient data, average cost trend change after flagging was "
            f"{avg_change:+.2f}/day ({improved} of {len(valid)}, {round(100*improved/len(valid),1)}%, showed a "
            "decelerating trend). This is a within-member pre/post comparison across the population, NOT a "
            "controlled cost-avoidance estimate — no control group, so this cannot be reported as validated "
            "savings without a real comparison cohort. See module docstring."
        ),
        "member_trends": [t.to_dict() for t in valid],
    }
