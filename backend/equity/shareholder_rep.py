"""Shareholder representative: consent tracking and sale-proceeds allocation (escrow, holdback, expense fund).

For a sale of the company, a stockholder representative acts for all holders after closing. This module tracks required
approvals and shows each holder's gross, withheld and net proceeds. It records who the representative is. Everlyn Ndirangu, the CFO, is the appointed representative; the formal appointment is documented in the sale agreement.
"""
from __future__ import annotations

from .cap_table import CapTable

REP = {"name": "Everlyn Ndirangu", "title": "Chief Financial Officer", "appointed_by": "The company (the CFO acts while the company is an early-stage startup)", "authority": ["Negotiate and agree purchase-price adjustments", "Receive and act on indemnification claims",
                "Control the escrow and the expense fund", "Settle disputes on behalf of the holders", "Give and receive notices"],
                "expense_fund_pct": 0.005, "successor": None,
       "status": "Appointed. To be written into the purchase agreement or a joinder at a sale, with a named successor."}


def required_approvals(t: CapTable, *, board: bool = True, class_vote_pct: float = 0.5) -> list[dict]:
    founder_pct = sum(t.common.values()) / t.common_total
    return [{"approval": "Board of Directors", "required": board, "threshold": "majority of directors"},
            {"approval": "Stockholders (DGCL 251 merger / 271 asset sale)", "required": True,
             "threshold": f"majority of outstanding voting stock (sole stockholder holds {founder_pct:.0%})"},
            {"approval": "Option holders (acceleration / assumption notices)", "required": True, "threshold": "notice; consent per plan and award terms"},
            {"approval": "SAFE holders (Mayo Clinic Platform, Dr. Nahar)", "required": True, "threshold": "per the SAFE: cash-out or conversion on a liquidity event"}]


def consent_status(holders: dict[str, float], signed: set[str], threshold: float = 0.5) -> dict:
    tot = sum(holders.values())
    got = sum(v for h, v in holders.items() if h in signed)
    return {"signed_pct": round(got / tot, 6) if tot else 0, "threshold": threshold, "met": got / tot > threshold if tot else False,
            "outstanding": [h for h in holders if h not in signed]}


def allocate(payouts: list[dict], *, escrow_pct: float = 0.10, expense_fund: float = 0.0, holdback_pct: float = 0.0) -> dict:
    total = sum(p["amount"] for p in payouts)
    rows = []
    for p in payouts:
        share = p["amount"] / total if total else 0
        escrow, hold, exp = p["amount"] * escrow_pct, p["amount"] * holdback_pct, expense_fund * share
        rows.append({"holder": p["holder"], "gross": round(p["amount"], 2), "escrow": round(escrow, 2), "holdback": round(hold, 2),
                     "expense_fund": round(exp, 2), "paid_at_closing": round(p["amount"] - escrow - hold - exp, 2)})
    return {"rows": rows, "totals": {k: round(sum(r[k] for r in rows), 2) for k in ("gross", "escrow", "holdback", "expense_fund", "paid_at_closing")}}
