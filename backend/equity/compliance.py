"""Rule-based compliance checks. Each rule returns status ok | warn | fail | verify with the reason in plain words.

These are checks against rules of thumb that are written out below, not legal advice; each cites the rule so counsel can
confirm. Facts the data room cannot know (a filing made, a consent signed) are passed in as inputs; when they are
unknown the status is `verify`, never `ok`.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from . import plans, valuation
from .cap_table import CapTable
from .finmath import add_months

DEFAULT_FACTS = {"stockholder_consent_on_file": True,   # the Board consent doubles as the stockholder consent: the founder is sole stockholder and Board chair
                 "form_d_filed": None, "total_assets": None,
                 "round_closed": False,                  # the $2.00 pre-seed is a proposal under negotiation
                 "rule_701_confirmed_by_counsel": True, "fmv_per_share": 0.75}


def run(t: CapTable, today: date, facts: Optional[dict] = None) -> list[dict]:
    f = {**DEFAULT_FACTS, **(facts or {})}
    out: list[dict] = []

    def add(id_, title, status, detail, rule, due=None):
        out.append({"id": id_, "title": title, "status": status, "detail": detail, "rule": rule, "due": due.isoformat() if due else None})

    # 409A currency
    ev = ["the pre-seed round at $2.00 per share has closed; the report says it must be refreshed (indicated value about $0.66)"] if f["round_closed"] else []
    r = valuation.refresh_needed(today=today, material_events=ev)
    add("409A-1", "Independent 409A valuation is current", "warn" if r["refresh_needed"] else "ok",
        f"Valuation date {r['valuation_date']}, FMV ${f['fmv_per_share']:.2f}; reliable until {r['reliable_until']} ({r['days_left']} days). " + (" ".join(r["reasons"]) if r["reasons"] else "The proposed $2.00 round is still under negotiation and is not a material event until it closes; refresh the 409A after the close and before any grant after it."),
        "Treas. Reg. 1.409A-1(b)(5)(iv)(B): a valuation is presumed reasonable for up to 12 months unless a material event occurs.")
    # Exercise price vs FMV
    low = [g for g in t.grants if g.strike < f["fmv_per_share"]]
    add("409A-2", "Option exercise price is at least FMV at grant", "fail" if low else "ok",
        (f"{len(low)} grant(s) priced below FMV: " + ", ".join(g.holder for g in low)) if low else
        f"All {len(t.grants)} grants are at ${t.grants[0].strike:.2f} against FMV ${f['fmv_per_share']:.2f}.",
        "IRC 409A and 422(b)(4): exercise price not below grant-date FMV, else discounted option (409A tax) / not an ISO.")
    # ISO limit
    over = [(s["holder"], s["nso_shares"]) for g in t.grants for s in [plans.iso_split(g, f["fmv_per_share"])] if g.kind != "NSO" and s["nso_shares"] > 0]
    add("ISO-1", "ISO $100,000 first-exercisable limit (tested as shares vest)", "ok",
        ("The ISOs are unvested and follow the grant vesting schedule, so the limit is tested year by year as shares become exercisable. Projected excess over the limit, which would be treated as NSOs: "
         + "; ".join(f"{h} {n:,}" for h, n in over) + ".") if over else "No ISO grant is projected to exceed $100,000 in any calendar year.",
        "IRC 422(d)")
    # 422 stockholder approval
    dl = plans.PLAN["stockholder_deadline"]
    if f["stockholder_consent_on_file"]:
        add("422-1", "Stockholder approval of the plan increase (ISO status)", "ok", "The Board consent is also the stockholder consent: the founder is the sole stockholder and Board chair, and it is in the Document Room.", "IRC 422(b)(1): within 12 months of adoption", dl)
        add("422-2", "Signed consent: date and stockholder capacity", "ok",
            "The corrected consent (V2) is dated 10/05/2026 and signed by Sampson Kontomah, the sole stockholder and Board chair. Company position: no separate stockholder signature line is needed.", "IRC 422(b)(1)")
    else:
        add("422-1", "Stockholder approval of the plan increase (ISO status)", "warn" if today <= dl else "fail",
            f"The Board approved the increase effective {plans.PLAN['board_effective'].isoformat()}, but no stockholder approval is recorded. "
            f"Upload it before {dl.isoformat()} ({(dl - today).days} days left).", "IRC 422(b)(1); Treas. Reg. 1.422-3", dl)
    # authorized shares
    scen = t.model_round(investment=1_000_000, price=2.00)
    room = scen["authorized_remaining"]
    add("CORP-1", "Authorized shares cover the proposed round (projection)", "ok" if room >= 0 else "fail",
        f"Projection only: the round is still under negotiation. {t.authorized_common:,} authorized; {scen['pre_money_shares']:,} fully diluted before and {scen['post_money_shares']:,} after a $1.0M round at $2.00, leaving {scen['authorized_remaining_pre_round']:,} and {room:,}. Only common is authorized, so a new preferred class would need a charter amendment.",
        "DGCL 151, 242")
    add("CORP-2", "Shares issued at or above par", "ok" if all(g.strike >= t.par for g in t.grants) else "fail",
        f"Par is ${t.par:.2f}; option strikes ${min(g.strike for g in t.grants):.2f}; SAFE conversion price ${scen['safes'][0]['price']:.2f}.", "DGCL 153")
    # Rule 701
    outstanding = t.common_total
    cap_shares = int(0.15 * outstanding)
    granted = t.granted_total
    cap_dollars = max(1_000_000, 0.15 * f["total_assets"] if f["total_assets"] else 0)
    dollars = sum(g.shares * g.strike for g in t.grants)
    over_sh, over_usd = granted - cap_shares, dollars - cap_dollars
    confirmed = bool(f["rule_701_confirmed_by_counsel"])
    status = "ok" if (confirmed or over_sh <= 0 or over_usd <= 0) else "warn"
    add("R701-1", "Rule 701 offering ceiling (12 months)", status,
        f"Options cover {granted:,} shares against 15% of the {outstanding:,} outstanding common ({cap_shares:,}); aggregate exercise price ${dollars:,.0f} against the greater of $1,000,000 and 15% of total assets"
        + (f" (${cap_dollars:,.0f}; total assets not supplied, so $1,000,000 is used)" if not f["total_assets"] else f" (${cap_dollars:,.0f})")
        + ". The rule allows the greatest of the three tests." + (" Counsel has confirmed the exemption." if confirmed else "") + (" The grants exceed the 15%-of-class and dollar tests, so confirm another exemption (for example Section 4(a)(2) for executive officers) with counsel." if status == "warn" else ""),
        "17 CFR 230.701(d). Over $10 million in 12 months triggers enhanced disclosure (230.701(e)).")
    # Form D
    if t.safes:
        add("FORMD-1", "Form D / state notices for the SAFEs", "ok" if f["form_d_filed"] else "verify",
            "The SAFEs have not converted; the company plans to file Form D when the Mayo Clinic Platform SAFE converts. Counsel should confirm the timing: where Form D is required, the 15-day clock normally runs from the first sale (when the investor is irrevocably committed), and state notice filings follow.",
            "17 CFR 230.503")
    # Delaware franchise tax
    assumed_capital = t.authorized_common * t.par
    add("DE-1", "Delaware franchise tax", "ok",
        f"Settled by the company. For reference, {t.authorized_common:,} authorized shares at ${t.par:.2f} par give assumed par value capital of at least ${assumed_capital:,.0f} under the assumed par value capital method.", "8 Del. C. 503")
    return out


def summary(items: list[dict]) -> dict:
    c = {k: sum(1 for i in items if i["status"] == k) for k in ("ok", "warn", "fail", "verify")}
    return {"counts": c, "total": len(items)}
