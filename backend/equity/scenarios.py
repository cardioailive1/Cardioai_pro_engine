"""Funding scenarios: compare raises side by side (price, post-money, dilution, who owns what, authorized-share room)."""
from __future__ import annotations

from .cap_table import CapTable

DEFAULTS = [
    {"name": "Proposed pre-seed (under negotiation)", "investment": 1_000_000, "price": 2.00},
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


# ---- Seed round (stacked on the proposed pre-seed) ----------------------------------------------------------------
# From CardioAI_RevenueModel_V7A (Funding & Phasing sheet). The model marks the seed amount "illustrative - replace with the
# negotiated amount" and gives no seed valuation, so the pre-money values below are inputs to test, not terms.
SEED_CONTEXT = {
    "source": "CardioAI_RevenueModel_V7A, Funding & Phasing",
    "seed_amount": 7_000_000, "target_close": "2027-10-01", "amount_status": "Illustrative: replace with the negotiated amount",
    "pre_seed_runway_months": 12, "pre_seed_funds_from": "2026-10-01",
    "seed_runway_at_fy2028_opex_months": 5.0,
    "runway_note": "The model's own caution: annual figures hide the monthly cash low point; its 36-month monthly model will show the true minimum cash and the seed size needed.",
}


def seed_pre_moneys(t: CapTable, pre_seed_investment: float = 1_000_000, pre_seed_price: float = 2.00) -> list[tuple[str, float]]:
    flat = t.model_round(investment=pre_seed_investment, price=pre_seed_price)["post_money"]
    return [("Flat to the pre-seed post-money", flat), ("$50M pre-money", 50_000_000), ("$75M pre-money", 75_000_000)]


def seed(t: CapTable, *, seed_amount: float = SEED_CONTEXT["seed_amount"], pre_moneys: list[tuple[str, float]] | None = None,
         pool_increase: int = 0, pre_seed_investment: float = 1_000_000, pre_seed_price: float = 2.00) -> dict:
    """Pre-seed at its proposed price, then the seed at each pre-money. Everything after the pre-seed is a projection of two
    proposed rounds, neither closed."""
    founder = next(iter(t.common))
    pre = t.model_round(investment=pre_seed_investment, price=pre_seed_price, investor="Pre-seed investors")
    t2 = t.after_round(pre)
    today = {r["holder"]: r["pct"] for r in t.current_ownership()}
    out = []
    for name, pm in pre_moneys or seed_pre_moneys(t, pre_seed_investment, pre_seed_price):
        r = t2.model_round(investment=seed_amount, pre_money=pm, pool_increase=pool_increase, investor="Seed investors")
        pct = lambda cls: sum(x["pct"] for x in r["rows"] if x["class"] in cls)
        f = next(x["pct"] for x in r["rows"] if x["holder"] == founder)
        f_pre = next(x["pct"] for x in pre["rows"] if x["holder"] == founder)
        flags = []
        if r["price"] < pre_seed_price:
            flags.append("Down round: seed price is below the pre-seed price")
        if r["authorized_remaining"] < 0:
            flags.append("Exceeds authorized shares: charter amendment needed")
        out.append({"name": name, "seed_amount": seed_amount, "pre_money": r["pre_money"], "post_money": r["post_money"], "price": r["price"],
                    "step_up_vs_pre_seed": round(r["price"] / pre_seed_price, 4), "new_shares": r["new_shares"],
                    "seed_investor_pct": pct({"Preferred (new)"}), "pre_seed_investor_pct": next(x["pct"] for x in r["rows"] if x["holder"] == "Pre-seed investors"),
                    "converted_safes_pct": sum(x["pct"] for x in r["rows"] if "converted SAFE" in x["holder"]),
                    "founder_pct": f, "founder_pct_today": today[founder], "founder_pct_after_pre_seed": f_pre,
                    "founder_dilution_from_seed_pts": f_pre - f, "options_and_pool_pct": pct({"Options (granted)", "Pool"}),
                    "post_money_shares": r["post_money_shares"], "authorized_remaining": r["authorized_remaining"], "flags": flags})
    return {"context": SEED_CONTEXT, "pre_seed": {"investment": pre_seed_investment, "price": pre_seed_price, "post_money": pre["post_money"],
                                                   "post_money_shares": pre["post_money_shares"], "founder_pct": next(x["pct"] for x in pre["rows"] if x["holder"] == founder)},
            "scenarios": out, "pool_increase": pool_increase,
            "note": "Two proposed rounds, neither closed; every figure after today is a projection. Seed pre-money values are illustrative inputs."}
