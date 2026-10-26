"""Funding scenarios: compare raises side by side (price, post-money, dilution, who owns what, authorized-share room)."""
from __future__ import annotations

from .cap_table import CapTable

DEFAULTS = [
    {"name": "Planned pre-seed", "investment": 1_000_000, "price": 2.00},
    {"name": "Larger pre-seed", "investment": 2_000_000, "price": 2.00},
    {"name": "Lower price", "investment": 1_000_000, "price": 1.60},
    {"name": "Seed on a $20M pre-money", "investment": 3_000_000, "pre_money": 20_000_000, "pool_increase": 1_000_000},
]


def compare(t: CapTable, scenarios: list[dict] | None = None) -> list[dict]:
    out = []
    founder = next(iter(t.common))
    today = {r["holder"]: r["pct"] for r in t.current_ownership()}
    for s in scenarios or DEFAULTS:
        r = t.model_round(investment=s["investment"], price=s.get("price"), pre_money=s.get("pre_money"),
                          pool_increase=s.get("pool_increase", 0))
        by = {x["holder"]: x["pct"] for x in r["rows"]}
        safes = sum(x["pct"] for x in r["rows"] if x["class"] == "SAFE (converted)")
        pool = sum(x["pct"] for x in r["rows"] if x["class"] in ("Pool", "Options (granted)"))
        out.append({"name": s.get("name", "Scenario"), "investment": s["investment"], "price": r["price"],
                    "pre_money": r["pre_money"], "post_money": r["post_money"], "new_investor_pct": by["New investors"],
                    "founder_pct": by[founder], "founder_pct_today": today[founder], "founder_dilution_pts": today[founder] - by[founder],
                    "safes_pct": safes, "options_and_pool_pct": pool, "post_money_shares": r["post_money_shares"],
                    "authorized_remaining": r["authorized_remaining"],
                    "flags": (["Exceeds authorized shares: charter amendment needed"] if r["authorized_remaining"] < 0 else [])})
    return out
