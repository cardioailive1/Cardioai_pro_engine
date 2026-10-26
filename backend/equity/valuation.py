"""Company valuation, an indicative 409A-style common-stock value (OPM) and ASC 820 Level 3 SAFE fair value.

INDICATIVE ONLY. The concluded fair market value of common stock for tax purposes is the independent appraiser's
($0.75 as of 18 Sep 2026). `opm_common_value` reproduces that method (option-pricing method with breakpoints, then a
discount for lack of marketability) so a change in the cap table or the equity value can be re-run, and
`refresh_needed` says when the report is stale. It does not create safe-harbor protection.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from .cap_table import CapTable
from .finmath import add_months, bs_call, finnerty_dlom
from . import waterfall as wf

REPORT = {"valuation_date": date(2026, 9, 18), "fmv_common": 0.75, "equity_value": 15_700_000.0, "dlom": 0.30,
          "volatility": 0.70, "term_years": 3.0, "risk_free": 0.0375,
          "approaches": {"SAFE backsolve": (10_600_000, 0.20), "Guideline public companies": (12_700_000, 0.15),
                         "Income (DCF scenarios)": (22_800_000, 0.40), "Cost": (10_500_000, 0.25)}}


# ---- approaches -------------------------------------------------------------------------------------------------
def blend(approaches: dict[str, tuple[float, float]]) -> dict:
    w = sum(x[1] for x in approaches.values())
    if abs(w - 1) > 1e-6:
        raise ValueError(f"weights must sum to 100% (got {w:.1%})")
    return {"equity_value": round(sum(v * x for v, x in approaches.values()), 2),
            "lines": [{"approach": k, "value": v, "weight": x, "contribution": round(v * x, 2)} for k, (v, x) in approaches.items()]}


def guideline_companies(metric: float, multiples: dict[str, float]) -> dict:
    vals = sorted(multiples.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"median_multiple": med, "mean_multiple": sum(vals) / n, "indicated_value_median": round(metric * med, 2),
            "range": [round(metric * vals[0], 2), round(metric * vals[-1], 2)]}


def dcf(cash_flows: list[float], *, discount_rate: float, terminal_growth: float | None = None,
        exit_multiple: float | None = None, terminal_metric: float | None = None) -> dict:
    """Year-end cash flows. Terminal value by Gordon growth on the last flow, or exit multiple x terminal_metric."""
    pv = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows))
    n = len(cash_flows)
    tv = 0.0
    if exit_multiple is not None and terminal_metric is not None:
        tv = exit_multiple * terminal_metric
    elif terminal_growth is not None and cash_flows:
        if discount_rate <= terminal_growth:
            raise ValueError("discount rate must exceed terminal growth")
        tv = cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** n if n else 0.0
    return {"pv_cash_flows": round(pv, 2), "terminal_value": round(tv, 2), "pv_terminal_value": round(pv_tv, 2), "equity_value": round(pv + pv_tv, 2)}


def cost_approach(items: dict[str, float]) -> dict:
    return {"equity_value": round(sum(items.values()), 2), "items": items}


# ---- OPM / 409A-style --------------------------------------------------------------------------------------------
def _tranches(claims: list[wf.Claim]):
    ps = sorted({0.0, *[c.strike for c in claims if c.kind == "option"],
                 *[c.pref / c.shares for c in claims if c.kind == "preferred" and c.shares and not c.participating]})
    bps = [wf.total_at(claims, p) for p in ps]
    prefs = sum(c.pref for c in claims if c.kind == "preferred")
    out = []
    if prefs > 0:   # first tranche: equity from 0 up to the total preference goes to the preferred / SAFEs only
        out.append({"lower": 0.0, "upper": bps[0], "fractions": {c.name: (c.pref / prefs if c.kind == "preferred" else 0.0) for c in claims}})
    for i, p0 in enumerate(ps):
        last = i == len(ps) - 1
        p1 = p0 + 1.0 if last else ps[i + 1]
        den = wf.total_at(claims, p1) - bps[i]
        frac = {c.name: (wf._pay(c, p1) - wf._pay(c, p0)) / den for c in claims}
        out.append({"lower": bps[i], "upper": None if last else bps[i + 1], "fractions": frac})
    return out


def opm_common_value(t: CapTable, *, equity_value: float = REPORT["equity_value"], term_years: float = REPORT["term_years"],
                     volatility: float = REPORT["volatility"], risk_free: float = REPORT["risk_free"],
                     dlom: Optional[float] = REPORT["dlom"], round_price: float = 2.00) -> dict:
    claims = wf.claims_from_cap_table(t, round_price=round_price)
    tr = _tranches(claims)
    value = {c.name: 0.0 for c in claims}
    rows = []
    for k in tr:
        hi = 0.0 if k["upper"] is None else bs_call(equity_value, k["upper"], term_years, risk_free, volatility)
        v = bs_call(equity_value, k["lower"], term_years, risk_free, volatility) - hi
        rows.append({"from": round(k["lower"], 2), "to": None if k["upper"] is None else round(k["upper"], 2), "value": round(v, 2)})
        for n, f in k["fractions"].items():
            value[n] += v * f
    founder = next(c for c in claims if c.kind == "common")
    marketable = value[founder.name] / founder.shares
    d = finnerty_dlom(volatility, term_years) if dlom is None else dlom
    return {"equity_value": equity_value, "marketable_value_per_share": round(marketable, 4), "dlom": round(d, 4),
            "common_fmv_per_share": round(marketable * (1 - d), 4), "tranches": rows,
            "claim_values": {k: round(v, 2) for k, v in value.items()}, "sum_check": round(sum(value.values()), 2),
            "indicative": True,
            "note": "Reproduces the appraiser's OPM method. Not a 409A valuation; the concluded value is the appraiser's."}


def refresh_needed(*, today: date, valuation_date: date = REPORT["valuation_date"], material_events: list[str] | None = None) -> dict:
    expiry = add_months(valuation_date, 12)
    reasons = []
    if today > expiry:
        reasons.append(f"The report is older than 12 months (expired {expiry.isoformat()}).")
    for e in material_events or []:
        reasons.append(f"Material event: {e}.")
    return {"valuation_date": valuation_date.isoformat(), "reliable_until": expiry.isoformat(),
            "days_left": (expiry - today).days, "refresh_needed": bool(reasons), "reasons": reasons}


# ---- ASC 820 ----------------------------------------------------------------------------------------------------
def safe_fair_value(*, purchase: float, conversion_shares: float, price_if_financing: float, years_to_financing: float,
                    exit_equity_value: float, shares_fully_diluted: float, years_to_exit: float, discount_rate: float,
                    p_financing: float, p_liquidity: float, p_dissolution: float, dissolution_recovery: float = 0.0) -> dict:
    """Scenario-weighted fair value of a SAFE (ASC 820 Level 3). Every probability and the discount rate are management inputs
    and are the unobservable inputs to disclose."""
    if abs(p_financing + p_liquidity + p_dissolution - 1) > 1e-6:
        raise ValueError("scenario probabilities must sum to 100%")
    fin = conversion_shares * price_if_financing / (1 + discount_rate) ** years_to_financing
    liq_share = conversion_shares / shares_fully_diluted * exit_equity_value
    liq = max(purchase, liq_share) / (1 + discount_rate) ** years_to_exit
    dis = purchase * dissolution_recovery
    fv = p_financing * fin + p_liquidity * liq + p_dissolution * dis
    return {"fair_value": round(fv, 2), "purchase_price": purchase, "premium_to_cost": round(fv / purchase - 1, 4),
            "scenarios": [{"name": "Equity financing (converts)", "probability": p_financing, "pv": round(fin, 2)},
                          {"name": "Liquidity event", "probability": p_liquidity, "pv": round(liq, 2)},
                          {"name": "Dissolution", "probability": p_dissolution, "pv": round(dis, 2)}],
            "level": 3, "unobservable_inputs": {"discount_rate": discount_rate, "p_financing": p_financing, "p_liquidity": p_liquidity,
                                                 "p_dissolution": p_dissolution, "years_to_financing": years_to_financing},
            "indicative": True}


def level3_rollforward(*, opening: float, purchases: float = 0.0, settlements_or_conversions: float = 0.0, fair_value_change: float = 0.0) -> dict:
    closing = opening + purchases - settlements_or_conversions + fair_value_change
    return {"opening": round(opening, 2), "purchases": round(purchases, 2), "settlements_or_conversions": round(-settlements_or_conversions, 2),
            "total_gains_losses_in_earnings": round(fair_value_change, 2), "closing": round(closing, 2)}
