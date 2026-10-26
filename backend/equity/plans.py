"""Equity plan administration: vesting, plan reserve, ISO $100,000 limit split."""
from __future__ import annotations

from datetime import date

from .cap_table import CapTable, Grant
from .finmath import add_months, months_between

PLAN = {"name": "2026 Equity Incentive Plan", "reserve_before": 1_108_696, "reserve_after": 3_108_696,
        "board_effective": date(2026, 10, 5), "stockholder_deadline": date(2027, 10, 5), "plan_term_years": 10}


def vested_shares(g: Grant, when: date) -> int:
    if g.fully_vested_at_grant:
        return g.shares
    m = months_between(g.grant_date, when)
    if m < g.cliff_months:
        return 0
    return g.shares if m >= g.vest_months else int(g.shares * m / g.vest_months)


def vesting_events(g: Grant) -> list[tuple[date, int]]:
    """(date, shares first exercisable). Cliff vests cliff/vest_months at once, then monthly."""
    if g.fully_vested_at_grant:
        return [(g.grant_date, g.shares)]
    ev, prev = [], 0
    for m in range(g.cliff_months, g.vest_months + 1):
        cum = g.shares if m == g.vest_months else int(g.shares * m / g.vest_months)
        ev.append((add_months(g.grant_date, m), cum - prev))
        prev = cum
    return ev


def iso_split(g: Grant, fmv_at_grant: float, limit: float = 100_000.0) -> dict:
    """IRC 422(d): shares first exercisable in a calendar year are ISOs only up to $100,000 of grant-date FMV; the rest are NSOs."""
    if g.kind == "NSO":
        return {"holder": g.holder, "iso_shares": 0, "nso_shares": g.shares, "by_year": {}}
    by: dict[int, int] = {}
    for d, n in vesting_events(g):
        by[d.year] = by.get(d.year, 0) + n
    iso = nso = 0
    rows = {}
    for y, n in sorted(by.items()):
        room = int(limit // fmv_at_grant) if fmv_at_grant > 0 else n
        i = min(n, room)
        iso += i
        nso += n - i
        rows[y] = {"first_exercisable": n, "fmv_value": round(n * fmv_at_grant, 2), "iso": i, "nso": n - i}
    return {"holder": g.holder, "iso_shares": iso, "nso_shares": nso, "by_year": rows}


def summary(t: CapTable, when: date, fmv_at_grant: float = 0.75) -> dict:
    rows = []
    for g in t.grants:
        v = vested_shares(g, when)
        nxt = next((d for d, _ in vesting_events(g) if d > when), None)
        rows.append({"holder": g.holder, "role": g.role, "type": g.kind, "shares": g.shares, "strike": g.strike,
                     "grant_date": g.grant_date.isoformat(), "vested": v, "unvested": g.shares - v,
                     "pct_vested": round(v / g.shares, 4), "next_vest": nxt.isoformat() if nxt else None,
                     "expires": add_months(g.grant_date, int(g.term_years * 12)).isoformat(),
                     "iso_nso": iso_split(g, fmv_at_grant)})
    return {"plan": PLAN["name"], "reserve": t.pool_reserve, "granted": t.granted_total, "available": t.pool_available,
            "reserve_increase": PLAN["reserve_after"] - PLAN["reserve_before"],
            "pct_granted": round(t.granted_total / t.pool_reserve, 4), "grants": rows,
            "totals": {"vested": sum(r["vested"] for r in rows), "unvested": sum(r["unvested"] for r in rows)}}
