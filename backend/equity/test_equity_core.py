from datetime import date

from equity import scenarios, waterfall
from equity.cap_table import seed_cap_table
from equity.finmath import bs_call, add_months, xirr


def test_pool_and_grants():
    t = seed_cap_table()
    assert t.granted_total == 2_034_457 and t.pool_available == 1_074_239


def test_safe_conversion_matches_cap_table():
    t = seed_cap_table()
    c = t.convert_safes(2.00)
    assert sum(x["shares"] for x in c) == 518_750
    assert [x["shares"] for x in c] == [500_000, 18_750]
    assert all(x["basis"] == "discount" and x["price"] == 1.6 for x in c)


def test_priced_round_matches_workbook():
    r = seed_cap_table().model_round(investment=1_000_000, price=2.00)
    assert r["pre_money_shares"] == 16_377_446 and r["post_money_shares"] == 16_877_446
    assert round(r["pre_money"]) == 32_754_892 and round(r["post_money"]) == 33_754_892
    assert r["authorized_remaining_pre_round"] == 8_622_554 and r["authorized_remaining"] == 8_122_554
    assert abs(sum(x["pct"] for x in r["rows"]) - 1) < 1e-9


def test_round_by_pre_money_solves_the_same_price():
    t = seed_cap_table()
    r = t.model_round(investment=1_000_000, pre_money=32_754_892)
    assert abs(r["price"] - 2.00) < 1e-4


def test_cap_binds_when_round_price_is_high():
    c = seed_cap_table().convert_safes(10.0)
    assert all(x["basis"] == "cap" for x in c) and c[0]["price"] < 8.0


def test_waterfall_breakpoints_match_409a_opm():
    bp = waterfall.breakpoints(waterfall.claims_from_cap_table(seed_cap_table()))
    vals = [b["equity_value"] for b in bp]
    assert abs(vals[0] - 830_000) < 1
    assert abs(vals[1] / 1e6 - 13.58) < 0.01                      # options in the money at $1.00
    assert abs(vals[-1] / 1e6 - 22.45) < 0.1                      # SAFEs convert at $1.60
    assert len(bp) == 3                                           # same-price holders are grouped into one row
    assert all(n in bp[1]["event"] for n in ("Everlyn", "Galax", "Avi", "Nahar"))
    assert "Mayo" in bp[2]["event"] and "Nahar" in bp[2]["event"]


def test_waterfall_conserves_value_and_orders_tiers():
    cl = waterfall.claims_from_cap_table(seed_cap_table())
    low = waterfall.distribute(cl, 500_000)
    assert low["total_paid"] == 500_000 and all(p["amount"] == 0 for p in low["payouts"] if "SAFE" not in p["holder"])
    for v in (5e6, 14e6, 30e6, 100e6):
        d = waterfall.distribute(cl, v)
        assert abs(d["total_paid"] - v) < 1
    hi = waterfall.distribute(cl, 100e6)
    saf = [p for p in hi["payouts"] if "SAFE" in p["holder"]]
    assert abs(sum(p["amount"] for p in saf) / 100e6 - 518_750 / (12_750_000 + 2_034_457 + 518_750)) < 1e-3


def test_participating_preferred_gets_more():
    from equity.waterfall import Claim, distribute
    common = Claim("Founder", "common", 1_000_000)
    np_ = distribute([common, Claim("A", "preferred", 500_000, pref=1e6)], 10e6)
    pp = distribute([common, Claim("A", "preferred", 500_000, pref=1e6, participating=True)], 10e6)
    a = lambda d: [p for p in d["payouts"] if p["holder"] == "A"][0]["amount"]
    assert a(pp) > a(np_)


def test_scenarios_flag_authorized_shortfall():
    out = scenarios.compare(seed_cap_table(), [{"name": "huge", "investment": 50e6, "price": 0.5}])
    assert out[0]["flags"]


def test_helpers():
    assert abs(bs_call(0.75, 1.0, 6.25, 0.039, 0.70) - 0.46) < 0.01
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert abs(xirr([(date(2024, 1, 1), -100), (date(2025, 1, 1), 200)]) - 1.0) < 0.01


def test_seed_stacks_on_proposed_pre_seed():
    t = seed_cap_table()
    s = scenarios.seed(t)
    assert s["context"]["seed_amount"] == 7_000_000 and s["pre_seed"]["post_money_shares"] == 16_877_446
    flat, p50, p75 = s["scenarios"]
    assert abs(flat["price"] - 2.00) < 1e-3 and abs(flat["step_up_vs_pre_seed"] - 1) < 1e-3     # flat to pre-seed post-money = same price
    assert abs(flat["post_money"] - (33_754_892 + 7_000_000)) < 50
    assert p75["price"] > p50["price"] > flat["price"]
    for r in s["scenarios"]:
        assert abs(r["founder_pct"] + r["seed_investor_pct"] + r["pre_seed_investor_pct"] + r["converted_safes_pct"] + r["options_and_pool_pct"] - 1) < 1e-9
        assert r["founder_pct"] < s["pre_seed"]["founder_pct"]
    # the SAFEs convert once, in the pre-seed (518,750 shares), not again in the seed
    assert round(flat["converted_safes_pct"] * flat["post_money_shares"]) == 518_750


def test_seed_flags_down_round_and_pool_increase():
    t = seed_cap_table()
    d = scenarios.seed(t, pre_moneys=[("low", 20_000_000)])["scenarios"][0]
    assert d["price"] < 2.0 and any("Down round" in f for f in d["flags"])
    base = scenarios.seed(t, pre_moneys=[("x", 50_000_000)])["scenarios"][0]
    bigger = scenarios.seed(t, pre_moneys=[("x", 50_000_000)], pool_increase=1_000_000)["scenarios"][0]
    assert bigger["options_and_pool_pct"] > base["options_and_pool_pct"]
