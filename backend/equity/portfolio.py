"""Investor portfolio view: each investor's cost, as-converted ownership, mark, MOIC, IRR and exit proceeds."""
from __future__ import annotations

from datetime import date

from .cap_table import CapTable
from .finmath import xirr
from . import waterfall as wf


def holdings(t: CapTable, *, mark_price: float = 2.00, mark_date: date, round_price: float = 2.00, exit_value: float | None = None) -> dict:
    conv = t.convert_safes(round_price)
    post = t.model_round(investment=0.0001, price=round_price)["post_money_shares"]
    dist = None
    if exit_value is not None:
        dist = {p["holder"]: p["amount"] for p in wf.distribute(wf.claims_from_cap_table(t, round_price=round_price), exit_value)["payouts"]}
    rows = []
    for s, c in zip(t.safes, conv):
        val = c["shares"] * mark_price
        flows = [(s.date, -s.purchase), (mark_date, val)] if s.date else None
        moic = val / s.purchase
        irr = xirr(flows) if flows and mark_date > s.date else None
        rows.append({"investor": s.holder, "instrument": "Post-money SAFE", "invested": s.purchase, "date": s.date.isoformat() if s.date else None,
                     "conversion_price": c["price"], "shares": c["shares"], "ownership": round(c["shares"] / post, 6), "mark_price": mark_price,
                     "mark_value": round(val, 2), "unrealized_gain": round(val - s.purchase, 2), "moic": round(moic, 4),
                     "irr": None if irr is None else round(irr, 4),
                     "exit_proceeds": None if dist is None else dist.get(f"{s.holder} (SAFE)")})
    ti = sum(r["invested"] for r in rows)
    tv = sum(r["mark_value"] for r in rows)
    return {"mark_date": mark_date.isoformat(), "mark_price": mark_price, "holdings": rows,
            "total": {"invested": ti, "mark_value": round(tv, 2), "moic": round(tv / ti, 4) if ti else None},
            "note": "Marks are indicative: the SAFE converts at a 20% discount, so the mark is the next round price times the conversion shares. "
                    "An ASC 820 fair value for reporting should use valuation.safe_fair_value with documented inputs."}
