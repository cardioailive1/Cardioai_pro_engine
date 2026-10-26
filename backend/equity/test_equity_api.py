import os
os.environ.setdefault("APP_ENCRYPTION_KEY", "x" * 43 + "=")
from datetime import date

from fastapi import FastAPI
from fastapi.testclient import TestClient

from equity import asc718, compliance, insights, plans, portfolio, shareholder_rep, valuation
from equity.api import router
from equity.cap_table import seed_cap_table

app = FastAPI()
app.include_router(router, prefix="/api")
c = TestClient(app)
B = "/api/data-room/equity"


def test_asc718_matches_report():
    r = asc718.run(seed_cap_table())
    assert abs(r["total_expense"] - 970_000) < 3_000
    g = {x["holder"]: x for x in r["grants"]}
    assert abs(g["Dr. Tamanna Nahar"]["total_expense"] - 223_000) < 2_000
    assert abs(g["Galax Womack"]["fair_value_per_option"] - 0.4575) < 0.002


def test_asc718_journal_balances_and_recognition():
    t = seed_cap_table()
    j = asc718.journal_entry(t, date(2026, 10, 1), date(2026, 12, 31))
    assert j["lines"][0]["debit"] == j["lines"][1]["credit"] > 0
    a = asc718.as_of(t, date(2030, 12, 31))
    assert abs(a["recognised_to_date"] - a["total_expense"]) < 1 and a["unrecognised"] < 1


def test_opm_reproduces_409a():
    v = valuation.opm_common_value(seed_cap_table())
    assert abs(v["marketable_value_per_share"] - 1.07) < 0.005 and abs(v["common_fmv_per_share"] - 0.75) < 0.005
    assert abs(v["sum_check"] - v["equity_value"]) < 1
    assert abs(valuation.blend(valuation.REPORT["approaches"])["equity_value"] / 1e6 - 15.77) < 0.01


def test_blend_rejects_bad_weights():
    r = c.post(f"{B}/valuation/blend", json={"approaches": {"a": [1e6, 0.5]}})
    assert r.status_code == 422


def test_refresh_rules():
    r = valuation.refresh_needed(today=date(2027, 10, 1))
    assert r["refresh_needed"] and "older than 12 months" in r["reasons"][0]
    assert not valuation.refresh_needed(today=date(2026, 10, 26))["refresh_needed"]


def test_asc820_safe_probabilities_validated():
    base = dict(exit_equity_value=50e6, discount_rate=0.25, p_financing=0.6, p_liquidity=0.3, p_dissolution=0.1)
    r = c.post(f"{B}/valuation/asc820-safe", json=base)
    assert r.status_code == 200 and r.json()["level"] == 3
    assert c.post(f"{B}/valuation/asc820-safe", json={**base, "p_dissolution": 0.5}).status_code == 422


def test_iso_limit_and_vesting():
    t = seed_cap_table()
    s = plans.iso_split(t.grants[0], 0.75)
    assert s["iso_shares"] + s["nso_shares"] == 594_348 and s["nso_shares"] > 0
    assert plans.iso_split(t.grants[3], 0.75)["iso_shares"] == 0
    assert plans.vested_shares(t.grants[3], date(2026, 10, 26)) == 400_000
    assert plans.vested_shares(t.grants[0], date(2027, 10, 4)) == 0
    assert plans.vested_shares(t.grants[0], date(2027, 10, 5)) == 148_587
    assert plans.vested_shares(t.grants[0], date(2030, 10, 5)) == 594_348


def test_compliance_follows_company_facts():
    d = date(2026, 10, 26)
    items = {i["id"]: i for i in compliance.run(seed_cap_table(), d)}
    assert items["409A-1"]["status"] == "ok" and "under negotiation" in items["409A-1"]["detail"]   # round not closed: no refresh yet
    assert items["422-1"]["status"] == "ok" and items["422-2"]["status"] == "ok"             # Board consent doubles as stockholder consent
    assert items["R701-1"]["status"] == "ok" and "Counsel has confirmed" in items["R701-1"]["detail"]
    assert items["ISO-1"]["status"] == "ok" and "Projected excess" in items["ISO-1"]["detail"]
    assert items["FORMD-1"]["status"] == "verify"                                                 # unknown filing is never reported as ok
    assert items["409A-2"]["status"] == "ok" and "projection" in items["CORP-1"]["title"].lower()
    closed = {i["id"]: i for i in compliance.run(seed_cap_table(), d, {"round_closed": True})}
    assert closed["409A-1"]["status"] == "warn"                                                   # refresh needed once the round closes
    nocons = {i["id"]: i for i in compliance.run(seed_cap_table(), d, {"stockholder_consent_on_file": False})}
    assert nocons["422-1"]["status"] == "warn"
    late = {i["id"]: i for i in compliance.run(seed_cap_table(), date(2027, 11, 1), {"stockholder_consent_on_file": False})}
    assert late["422-1"]["status"] == "fail"
    assert {i["id"]: i for i in compliance.run(seed_cap_table(), d, {"rule_701_confirmed_by_counsel": False})}["R701-1"]["status"] == "warn"


def test_checklist_moves_with_facts():
    t = seed_cap_table()
    a = insights.checklist(t, date(2026, 10, 26))["score_pct"]
    assert insights.checklist(t, date(2026, 10, 26), {"rep_named": True})["score_pct"] > a
    assert insights.checklist(t, date(2026, 10, 26), {"round_closed": True})["score_pct"] < a


def test_portfolio_and_rep():
    p = portfolio.holdings(seed_cap_table(), mark_date=date(2026, 10, 26), exit_value=30e6)
    assert p["total"]["invested"] == 830_000 and abs(p["total"]["moic"] - 1.25) < 1e-6
    t = seed_cap_table()
    assert shareholder_rep.consent_status({"a": 1, "b": 1}, {"a"}, 0.5)["met"] is False
    al = shareholder_rep.allocate([{"holder": "x", "amount": 100.0}], escrow_pct=0.1, expense_fund=1)
    assert al["rows"][0]["paid_at_closing"] == 89.0


def test_endpoints_ok_and_validated():
    for path in ("overview", "cap-table", "scenarios/default", "asc718", "valuation", "plans", "compliance", "portfolio", "shareholder-rep"):
        r = c.get(f"{B}/{path}")
        assert r.status_code == 200, path
        assert "not a 409a valuation" in r.json()["disclaimer"].lower()
    assert c.post(f"{B}/cap-table/round", json={"investment": 1e6}).status_code == 422      # neither price nor pre-money
    assert c.post(f"{B}/cap-table/round", json={"investment": -5, "price": 2}).status_code == 422
    r = c.post(f"{B}/waterfall", json={"equity_value": 30e6}).json()
    assert abs(r["distribution"]["total_paid"] - 30e6) < 1
    assert c.get(f"{B}/portfolio?mark_price=-1").status_code == 422
    r = c.post(f"{B}/shareholder-rep/allocate", json={"equity_value": 30e6, "escrow_pct": 0.1}).json()
    assert abs(r["totals"]["gross"] - 30e6) < 1
