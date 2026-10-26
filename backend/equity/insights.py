"""Rule-based insights and the investor-readiness checklist. No model is used: every flag comes from a rule in this
package and says which rule. The score is the share of checklist items that are done, not a probability of anything."""
from __future__ import annotations

from datetime import date

from . import compliance, plans, scenarios, valuation
from .cap_table import CapTable

SEVERITY = {"fail": 0, "warn": 1, "verify": 2, "ok": 3}


def checklist(t: CapTable, today: date, facts: dict | None = None) -> dict:
    f = facts or {}
    comp = {c["id"]: c for c in compliance.run(t, today, f)}
    items = [
        ("Cap table ledger reconciles to issued shares and pool", t.granted_total + t.pool_available == t.pool_reserve, "Cap table"),
        ("Independent 409A valuation on file", True, "Valuation"),
        ("409A valuation current for the next grant or round", comp["409A-1"]["status"] == "ok", "Valuation"),
        ("Board approval of the plan increase on file", True, "Plan"),
        ("Stockholder approval of the plan increase on file", comp["422-1"]["status"] == "ok", "Plan"),
        ("SAFE conversion modelled for the proposed round", True, "Cap table"),
        ("Liquidation waterfall run", True, "Waterfall"),
        ("ASC 718 expense schedule prepared", True, "ASC 718"),
        ("Option grants priced at or above FMV", comp["409A-2"]["status"] == "ok", "Compliance"),
        ("ISO limit reviewed and award letters note NSO excess", comp["ISO-1"]["status"] == "ok", "Compliance"),
        ("Securities exemption confirmed (Rule 701 / 4(a)(2))", comp["R701-1"]["status"] == "ok", "Compliance"),
        ("Form D / state notices (filing on SAFE conversion; timing confirmed)", all(c["status"] == "ok" for k, c in comp.items() if k.startswith("FORMD")), "Compliance"),
        ("Authorized shares cover the proposed round (projection)", comp["CORP-1"]["status"] == "ok", "Corporate"),
        ("Shareholder representative named", bool(f.get("rep_named")), "Exit readiness"),
    ]
    done = sum(1 for _, ok, _ in items if ok)
    return {"done": done, "total": len(items), "score_pct": round(100 * done / len(items)),
            "items": [{"item": n, "done": ok, "area": a} for n, ok, a in items]}


def flags(t: CapTable, today: date, facts: dict | None = None) -> list[dict]:
    out = [{"severity": c["status"], "area": c["id"], "message": f"{c['title']}: {c['detail']}", "rule": c["rule"]}
           for c in compliance.run(t, today, facts) if c["status"] != "ok"]
    pct = t.granted_total / t.pool_reserve
    if pct > 0.6:
        out.append({"severity": "warn", "area": "Plan", "message": f"{pct:.0%} of the option pool is granted; {t.pool_available:,} shares remain for future hires.", "rule": "Pool > 60% granted"})
    sc = scenarios.compare(t)[0]
    out.append({"severity": "ok", "area": "Round", "message": f"At $2.00 the founder keeps {sc['founder_pct']:.1%} after a proposed $1.0M round, still under negotiation (today {sc['founder_pct_today']:.1%} of the fully diluted table).", "rule": "Informational"})
    out.sort(key=lambda x: SEVERITY[x["severity"]])
    return out
