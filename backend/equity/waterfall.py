"""Liquidation waterfall: who gets what at a given exit value, and the breakpoints between tiers.

Every claim pays a function of the value per common share `p`:
  common            shares * p
  option            shares * max(p - strike, 0)
  preferred / SAFE  non-participating: max(preference, conversion_shares * p)
                    participating:     pref + conv*p (optionally capped at cap_multiple * pref, or converting if better)
Total payout rises with p, so the exit value is solved for p by bisection. Below the total preference, money goes by
seniority (rank 0 first), pro rata within a rank. Taxes, transaction fees, escrow and debt are not modelled here; they
are inputs (`deductions`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cap_table import CapTable
from .finmath import bisect


@dataclass
class Claim:
    name: str
    kind: str                      # "common" | "option" | "preferred"
    shares: float                  # shares held, or conversion shares for preferred
    strike: float = 0.0
    pref: float = 0.0
    participating: bool = False
    cap_multiple: Optional[float] = None
    rank: int = 0                  # 0 = most senior


def _pay(c: Claim, p: float) -> float:
    if c.kind == "common":
        return c.shares * p
    if c.kind == "option":
        return c.shares * max(p - c.strike, 0.0)
    conv = c.shares * p
    if c.participating:
        capped = c.pref + conv
        if c.cap_multiple:
            capped = min(capped, c.cap_multiple * c.pref)
        return max(capped, conv)
    return max(c.pref, conv)


def total_at(claims: list[Claim], p: float) -> float:
    return sum(_pay(c, p) for c in claims)


def claims_from_cap_table(t: CapTable, *, round_price: float = 2.00) -> list[Claim]:
    claims = [Claim(h, "common", n) for h, n in t.common.items()]
    claims += [Claim(f"{g.holder} (options)", "option", g.shares, strike=g.strike) for g in t.grants]
    for c in t.convert_safes(round_price):
        claims.append(Claim(f"{c['holder']} (SAFE)", "preferred", c["shares"], pref=c["purchase"]))
    return claims


def distribute(claims: list[Claim], equity_value: float, deductions: float = 0.0) -> dict:
    net = max(equity_value - deductions, 0.0)
    prefs = sum(c.pref for c in claims if c.kind == "preferred")
    pay: dict[str, float] = {c.name: 0.0 for c in claims}
    if net <= prefs + 1e-9:
        left = net
        for rank in sorted({c.rank for c in claims if c.pref}):
            grp = [c for c in claims if c.kind == "preferred" and c.rank == rank]
            need = sum(c.pref for c in grp)
            give = min(left, need)
            for c in grp:
                pay[c.name] = give * c.pref / need if need else 0.0
            left -= give
        p = 0.0
    else:
        hi = net + max([c.strike for c in claims] + [1.0]) + 1.0
        p = bisect(lambda x: total_at(claims, x) - net, 0.0, hi, tol=1e-10)
        for c in claims:
            pay[c.name] = _pay(c, p)
    total = sum(pay.values())
    return {"equity_value": equity_value, "deductions": deductions, "net_proceeds": net, "price_per_common_share": p,
            "payouts": [{"holder": k, "amount": round(v, 2), "pct": (v / total if total else 0.0)} for k, v in pay.items()],
            "total_paid": round(total, 2)}


def breakpoints(claims: list[Claim]) -> list[dict]:
    """Equity values at which a new tier of holders starts to share: preference cleared, each option strike reached,
    each preferred converting. Holders that hit the same price are grouped into one row."""
    prefs = sum(c.pref for c in claims if c.kind == "preferred")
    out = [{"equity_value": round(prefs, 2), "price_per_share": 0.0, "event": "Preferences cleared; common starts to receive value"}]
    groups: dict[tuple[str, float], list[str]] = {}
    for c in claims:
        if c.kind == "option":
            groups.setdefault(("opt", round(c.strike, 6)), []).append(c.name.replace(" (options)", ""))
        elif c.kind == "preferred" and c.shares and not c.participating:
            groups.setdefault(("conv", round(c.pref / c.shares, 6)), []).append(c.name.replace(" (SAFE)", ""))
    for (kind, price), names in groups.items():
        label = (f"Options in the money (strike ${price:.2f}): " if kind == "opt" else f"Convert to common (above ${price:.4f}/share): ") + ", ".join(names)
        out.append({"equity_value": round(total_at(claims, price), 2), "price_per_share": price, "event": label})
    out.sort(key=lambda r: (r["equity_value"], r["event"]))
    return out


def exit_matrix(claims: list[Claim], values: list[float], deductions_pct: float = 0.0) -> list[dict]:
    rows = []
    for v in values:
        d = distribute(claims, v, deductions=v * deductions_pct)
        rows.append({"equity_value": v, "price_per_share": round(d["price_per_common_share"], 4),
                     "payouts": {p["holder"]: p["amount"] for p in d["payouts"]}})
    return rows
