"""HTTP API for the equity suite. Mounted at /api/data-room/equity, so the data-room access-code gate in main.py applies."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import asc718, compliance, insights, plans, portfolio, scenarios, shareholder_rep, valuation, waterfall
from .cap_table import seed_cap_table

router = APIRouter(prefix="/data-room/equity", tags=["equity"])

DISCLAIMER = ("Indicative analytics from deterministic, rule-based engines (no language model). Not a 409A valuation, an ASC 820 opinion, "
              "an audit or legal advice. The 409A fair market value of $0.75 per common share is the independent appraiser's.")


def _wrap(**kw) -> dict:
    return {**kw, "disclaimer": DISCLAIMER, "as_of": date.today().isoformat()}


def _bad(e: Exception):
    raise HTTPException(status_code=422, detail=str(e))


class RoundIn(BaseModel):
    investment: float = Field(gt=0, le=1e10)
    price: Optional[float] = Field(default=None, gt=0, le=1e6)
    pre_money: Optional[float] = Field(default=None, gt=0, le=1e12)
    pool_increase: int = Field(default=0, ge=0, le=100_000_000)
    name: str = Field(default="Scenario", max_length=80)


class ScenariosIn(BaseModel):
    scenarios: list[RoundIn] = Field(min_length=1, max_length=12)


class SeedIn(BaseModel):
    seed_amount: float = Field(default=7_000_000, gt=0, le=1e10)
    pre_moneys: list[float] = Field(min_length=1, max_length=6)
    pool_increase: int = Field(default=0, ge=0, le=100_000_000)
    pre_seed_investment: float = Field(default=1_000_000, gt=0, le=1e10)
    pre_seed_price: float = Field(default=2.0, gt=0, le=1e6)


class WaterfallIn(BaseModel):
    equity_value: float = Field(gt=0, le=1e12)
    deductions_pct: float = Field(default=0.0, ge=0, lt=1)
    round_price: float = Field(default=2.0, gt=0, le=1e6)


class Asc718In(BaseModel):
    fmv: float = Field(default=0.75, gt=0, le=1e4)
    volatility: float = Field(default=0.70, gt=0, le=3)
    risk_free: float = Field(default=0.039, ge=0, le=0.25)
    forfeiture_rate: float = Field(default=0.0, ge=0, lt=1)
    as_of: Optional[date] = None


class OpmIn(BaseModel):
    equity_value: float = Field(default=valuation.REPORT["equity_value"], gt=0, le=1e12)
    term_years: float = Field(default=3.0, gt=0, le=15)
    volatility: float = Field(default=0.70, gt=0, le=3)
    risk_free: float = Field(default=0.0375, ge=0, le=0.25)
    dlom: Optional[float] = Field(default=0.30, ge=0, lt=1)


class BlendIn(BaseModel):
    approaches: dict[str, tuple[float, float]]


class Safe820In(BaseModel):
    purchase: float = Field(default=830_000, gt=0)
    conversion_shares: float = Field(default=518_750, gt=0)
    price_if_financing: float = Field(default=2.0, gt=0)
    years_to_financing: float = Field(default=0.5, ge=0, le=20)
    exit_equity_value: float = Field(gt=0, le=1e12)
    shares_fully_diluted: float = Field(default=16_877_446, gt=0)
    years_to_exit: float = Field(default=5.0, gt=0, le=30)
    discount_rate: float = Field(gt=0, le=1)
    p_financing: float = Field(ge=0, le=1)
    p_liquidity: float = Field(ge=0, le=1)
    p_dissolution: float = Field(ge=0, le=1)
    dissolution_recovery: float = Field(default=0.0, ge=0, le=1)


class FactsIn(BaseModel):
    stockholder_consent_on_file: bool = True
    form_d_filed: Optional[bool] = None
    total_assets: Optional[float] = Field(default=None, ge=0)
    round_closed: bool = False
    rule_701_confirmed_by_counsel: bool = True
    rep_named: bool = False


class AllocIn(BaseModel):
    equity_value: float = Field(gt=0, le=1e12)
    escrow_pct: float = Field(default=0.10, ge=0, le=0.5)
    holdback_pct: float = Field(default=0.0, ge=0, le=0.5)
    expense_fund: float = Field(default=0.0, ge=0)
    deductions_pct: float = Field(default=0.0, ge=0, lt=1)


class ConsentIn(BaseModel):
    signed: list[str] = []
    threshold: float = Field(default=0.5, gt=0, lt=1)


@router.get("/overview")
def overview():
    t, today = seed_cap_table(), date.today()
    sc = scenarios.compare(t)[0]
    ck = insights.checklist(t, today)
    return _wrap(company=t.company, fmv_common=valuation.REPORT["fmv_common"], equity_value=valuation.REPORT["equity_value"],
                 fully_diluted_today=t.common_total + t.pool_reserve, pool={"reserve": t.pool_reserve, "granted": t.granted_total, "available": t.pool_available},
                 proposed_round=sc, asc718_total=asc718.run(t)["total_expense"], readiness=ck, flags=insights.flags(t, today))


@router.get("/cap-table")
def cap_table():
    t = seed_cap_table()
    return _wrap(table=t.to_dict(), ownership_today=t.current_ownership(), safe_conversion=t.convert_safes(2.00),
                 proposed_round=t.model_round(investment=1_000_000, price=2.00))


@router.post("/cap-table/round")
def round_model(body: RoundIn):
    try:
        return _wrap(**seed_cap_table().model_round(investment=body.investment, price=body.price, pre_money=body.pre_money, pool_increase=body.pool_increase))
    except ValueError as e:
        _bad(e)


@router.post("/scenarios")
def funding_scenarios(body: ScenariosIn):
    try:
        return _wrap(scenarios=scenarios.compare(seed_cap_table(), [s.model_dump(exclude_none=True) for s in body.scenarios]))
    except ValueError as e:
        _bad(e)


@router.get("/scenarios/default")
def default_scenarios():
    return _wrap(scenarios=scenarios.compare(seed_cap_table()))


@router.get("/scenarios/seed")
def seed_default():
    return _wrap(**scenarios.seed(seed_cap_table()))


@router.post("/scenarios/seed")
def seed_run(body: SeedIn):
    if any(p <= 0 or p > 1e12 for p in body.pre_moneys):
        raise HTTPException(status_code=422, detail="pre-money values must be positive")
    try:
        pms = [(f"${p / 1e6:,.1f}M pre-money", p) for p in body.pre_moneys]
        return _wrap(**scenarios.seed(seed_cap_table(), seed_amount=body.seed_amount, pre_moneys=pms, pool_increase=body.pool_increase,
                                      pre_seed_investment=body.pre_seed_investment, pre_seed_price=body.pre_seed_price))
    except ValueError as e:
        _bad(e)


@router.post("/waterfall")
def waterfall_run(body: WaterfallIn):
    t = seed_cap_table()
    cl = waterfall.claims_from_cap_table(t, round_price=body.round_price)
    d = waterfall.distribute(cl, body.equity_value, deductions=body.equity_value * body.deductions_pct)
    return _wrap(distribution=d, breakpoints=waterfall.breakpoints(cl),
                 matrix=waterfall.exit_matrix(cl, [5e6, 10e6, 15e6, 25e6, 50e6, 100e6, 250e6], body.deductions_pct))


@router.get("/asc718")
def asc718_get(fmv: float = 0.75, volatility: float = 0.70, risk_free: float = 0.039, forfeiture_rate: float = 0.0):
    p = Asc718In(fmv=fmv, volatility=volatility, risk_free=risk_free, forfeiture_rate=forfeiture_rate)
    return asc718_run(p)


@router.post("/asc718")
def asc718_run(body: Asc718In):
    t = seed_cap_table()
    when = body.as_of or date.today()
    kw = dict(fmv=body.fmv, volatility=body.volatility, risk_free=body.risk_free, forfeiture_rate=body.forfeiture_rate)
    res = asc718.as_of(t, when, **kw)
    q_start = date(when.year, 3 * ((when.month - 1) // 3) + 1, 1)
    return _wrap(**res, journal_entry=asc718.journal_entry(t, q_start, when, **kw))


@router.get("/valuation")
def valuation_get():
    t, today = seed_cap_table(), date.today()
    return _wrap(report={"valuation_date": valuation.REPORT["valuation_date"].isoformat(), "fmv_common": valuation.REPORT["fmv_common"],
                         "dlom": valuation.REPORT["dlom"], **valuation.blend(valuation.REPORT["approaches"])},
                 opm=valuation.opm_common_value(t),
                 refresh=valuation.refresh_needed(today=today), refresh_trigger="The 409A is refreshed after the proposed $2.00 pre-seed round closes (still under negotiation); the report indicates about $0.66 per common share at that price.")


@router.post("/valuation/opm")
def valuation_opm(body: OpmIn):
    try:
        return _wrap(**valuation.opm_common_value(seed_cap_table(), **body.model_dump()))
    except ValueError as e:
        _bad(e)


@router.post("/valuation/blend")
def valuation_blend(body: BlendIn):
    try:
        return _wrap(**valuation.blend(body.approaches))
    except ValueError as e:
        _bad(e)


@router.post("/valuation/asc820-safe")
def valuation_asc820(body: Safe820In):
    try:
        return _wrap(**valuation.safe_fair_value(**body.model_dump()))
    except ValueError as e:
        _bad(e)


@router.get("/plans")
def plans_get():
    return _wrap(**plans.summary(seed_cap_table(), date.today()))


@router.post("/compliance")
def compliance_run(body: FactsIn):
    items = compliance.run(seed_cap_table(), date.today(), body.model_dump())
    return _wrap(items=items, **compliance.summary(items))


@router.get("/compliance")
def compliance_get():
    items = compliance.run(seed_cap_table(), date.today())
    return _wrap(items=items, **compliance.summary(items))


@router.get("/portfolio")
def portfolio_get(mark_price: float = 2.0, exit_value: Optional[float] = None):
    if mark_price <= 0 or mark_price > 1e6 or (exit_value is not None and not 0 < exit_value <= 1e12):
        raise HTTPException(422, "mark_price and exit_value must be positive")
    return _wrap(**portfolio.holdings(seed_cap_table(), mark_price=mark_price, mark_date=date.today(), exit_value=exit_value))


@router.get("/shareholder-rep")
def rep_get():
    t = seed_cap_table()
    return _wrap(representative=shareholder_rep.REP_TEMPLATE, approvals=shareholder_rep.required_approvals(t),
                 holders={h: n for h, n in t.common.items()})


@router.post("/shareholder-rep/consents")
def rep_consents(body: ConsentIn):
    t = seed_cap_table()
    return _wrap(**shareholder_rep.consent_status({h: float(n) for h, n in t.common.items()}, set(body.signed), body.threshold))


@router.post("/shareholder-rep/allocate")
def rep_allocate(body: AllocIn):
    t = seed_cap_table()
    cl = waterfall.claims_from_cap_table(t)
    d = waterfall.distribute(cl, body.equity_value, deductions=body.equity_value * body.deductions_pct)
    return _wrap(distribution_price=d["price_per_common_share"],
                 **shareholder_rep.allocate(d["payouts"], escrow_pct=body.escrow_pct, expense_fund=body.expense_fund, holdback_pct=body.holdback_pct))


@router.post("/insights")
def insights_run(body: FactsIn):
    t, today = seed_cap_table(), date.today()
    f = body.model_dump()
    return _wrap(flags=insights.flags(t, today, f), readiness=insights.checklist(t, today, f))
