"""ASC 718 stock-compensation expense: grant-date fair value (Black-Scholes), expense schedule, journal entries.

Policy assumptions (change them in the request): service-only awards, straight-line over the requisite service period,
forfeitures recognised as they occur (forfeiture_rate=0), awards with no vesting condition expensed at grant.
Inputs the appraiser's report uses: FMV $0.75, volatility 70%, risk-free ~3.9%, 6.25-year simplified-method term for
employees, 10-year contractual term for the fully vested director/advisor grant. Not an audit opinion.
"""
from __future__ import annotations

from datetime import date

from .cap_table import CapTable, Grant
from .finmath import add_months, bs_call

DEFAULTS = {"fmv": 0.75, "volatility": 0.70, "risk_free": 0.039, "dividend_yield": 0.0, "forfeiture_rate": 0.0}


def grant_fair_value(g: Grant, *, fmv: float, volatility: float, risk_free: float, dividend_yield: float = 0.0) -> float:
    return bs_call(fmv, g.strike, g.expected_term_years, risk_free, volatility, dividend_yield)


def schedule(g: Grant, fair_value: float, forfeiture_rate: float = 0.0) -> list[dict]:
    """Monthly expense, straight-line over the service period; grants vested at grant are expensed on the grant date."""
    total = g.shares * fair_value * (1 - forfeiture_rate)
    if g.fully_vested_at_grant or g.vest_months <= 0:
        return [{"date": g.grant_date.isoformat(), "expense": round(total, 2)}]
    per = total / g.vest_months
    return [{"date": add_months(g.grant_date, m).isoformat(), "expense": round(per, 2)} for m in range(1, g.vest_months + 1)]


def run(t: CapTable, **kw) -> dict:
    p = {**DEFAULTS, **kw}
    grants = []
    all_rows: list[tuple[str, str, float]] = []
    for g in t.grants:
        fv = grant_fair_value(g, fmv=p["fmv"], volatility=p["volatility"], risk_free=p["risk_free"], dividend_yield=p["dividend_yield"])
        sch = schedule(g, fv, p["forfeiture_rate"])
        total = sum(r["expense"] for r in sch)
        grants.append({"holder": g.holder, "role": g.role, "shares": g.shares, "strike": g.strike, "term_years": g.expected_term_years,
                       "fair_value_per_option": round(fv, 4), "total_expense": round(total, 2),
                       "recognition": "at grant (fully vested)" if g.fully_vested_at_grant else f"{g.vest_months} months straight-line"})
        all_rows += [(g.holder, r["date"], r["expense"]) for r in sch]
    by_year: dict[str, float] = {}
    by_quarter: dict[str, float] = {}
    for _, d, x in all_rows:
        y, m = d[:4], int(d[5:7])
        by_year[y] = by_year.get(y, 0.0) + x
        q = f"{y}-Q{(m - 1) // 3 + 1}"
        by_quarter[q] = by_quarter.get(q, 0.0) + x
    return {"assumptions": p, "grants": grants, "total_expense": round(sum(g["total_expense"] for g in grants), 2),
            "by_year": {k: round(v, 2) for k, v in sorted(by_year.items())},
            "by_quarter": {k: round(v, 2) for k, v in sorted(by_quarter.items())}}


def as_of(t: CapTable, when: date, **kw) -> dict:
    """Expense recognised to date and still to come, plus the journal entry for a period."""
    res = run(t, **kw)
    rows = []
    p = {**DEFAULTS, **kw}
    for g in t.grants:
        fv = grant_fair_value(g, fmv=p["fmv"], volatility=p["volatility"], risk_free=p["risk_free"], dividend_yield=p["dividend_yield"])
        rows += [(g.holder, r["date"], r["expense"]) for r in schedule(g, fv, p["forfeiture_rate"])]
    rec = sum(x for _, d, x in rows if d <= when.isoformat())
    total = sum(x for _, _, x in rows)
    res["recognised_to_date"] = round(rec, 2)
    res["unrecognised"] = round(total - rec, 2)
    res["as_of"] = when.isoformat()
    return res


def journal_entry(t: CapTable, start: date, end: date, **kw) -> dict:
    """Entry for expense falling in (start, end]. Debit stock-based compensation expense, credit additional paid-in capital."""
    p = {**DEFAULTS, **kw}
    amt = 0.0
    for g in t.grants:
        fv = grant_fair_value(g, fmv=p["fmv"], volatility=p["volatility"], risk_free=p["risk_free"], dividend_yield=p["dividend_yield"])
        amt += sum(r["expense"] for r in schedule(g, fv, p["forfeiture_rate"]) if start.isoformat() < r["date"] <= end.isoformat())
    amt = round(amt, 2)
    return {"period": f"{start.isoformat()} to {end.isoformat()}", "amount": amt,
            "lines": [{"account": "Stock-based compensation expense", "debit": amt, "credit": 0.0},
                      {"account": "Additional paid-in capital - stock options", "debit": 0.0, "credit": amt}],
            "note": "Allocate the debit between R&D and G&A by each employee's function before posting."}
