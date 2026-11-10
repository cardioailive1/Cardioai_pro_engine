from datetime import date

import pytest

from training.mayo import mace_corroboration as mc

EV = date(2025, 3, 10)


def test_parse_result_and_range():
    assert mc.parse_result("46") == ("=", 46.0)
    assert mc.parse_result("<0.02") == ("<", 0.02)
    assert mc.parse_result("CANCELED") is None and mc.parse_result(None) is None
    assert mc.parse_upper_limit("<=15") == 15.0
    assert mc.parse_upper_limit("0.00 - 0.12") == 0.12
    assert mc.parse_upper_limit(None) is None


def test_troponin_elevated():
    assert mc.troponin_elevated("46", "<=15") is True
    assert mc.troponin_elevated("6", "<=10") is False
    assert mc.troponin_elevated("<0.02", "0.00 - 0.10") is False
    assert mc.troponin_elevated("0.40", "0.00 - 0.12") is True
    assert mc.troponin_elevated("CANCELED", "<=15") is None
    assert mc.troponin_elevated("46", None) is None


def test_mi_tiers():
    hi, lo = [(date(2025, 3, 11), "46", "<=15")], [(date(2025, 3, 11), "6", "<=15")]
    assert mc.corroborate("myocardial_infarction", EV, hi, [], [])[0] == mc.CORROBORATED
    assert mc.corroborate("myocardial_infarction", EV, [], [(date(2025, 3, 12),)], [])[0] == mc.CORROBORATED
    assert mc.corroborate("myocardial_infarction", EV, lo, [], [])[0] == mc.CONFLICT
    assert mc.corroborate("myocardial_infarction", EV, [(date(2025, 2, 1), "46", "<=15")], [], [])[0] == mc.CODE_ONLY
    assert mc.corroborate("myocardial_infarction", EV, [], [], [])[0] == mc.CODE_ONLY
    # an elevated result beats a normal one in the same window; a revascularization beats a normal troponin
    assert mc.corroborate("myocardial_infarction", EV, lo + hi, [], [])[0] == mc.CORROBORATED
    assert mc.corroborate("myocardial_infarction", EV, lo, [(EV,)], [])[0] == mc.CORROBORATED


def test_stroke_and_death_tiers():
    assert mc.corroborate("stroke", EV, [], [], [(date(2025, 3, 10),)])[0] == mc.CORROBORATED
    assert mc.corroborate("stroke", EV, [], [], [(date(2025, 3, 20),)])[0] == mc.CODE_ONLY
    assert mc.corroborate("all_cause_death", EV, [], [], [])[0] == mc.CORROBORATED
    with pytest.raises(ValueError):
        mc.corroborate("bogus", EV, [], [], [])


def test_tier_records_and_summary():
    events = [{"key": "a", "pcn": "1", "event_type": "myocardial_infarction", "event_date": EV},
              {"key": "b", "pcn": "2", "event_type": "myocardial_infarction", "event_date": EV},
              {"key": "c", "pcn": "3", "event_type": "stroke", "event_date": EV}]
    trop = [{"pcn": "1", "d": EV, "result_txt": "46", "normal_range_txt": "<=15"},
            {"pcn": "2", "d": EV, "result_txt": "<6", "normal_range_txt": "<=10"}]
    out = mc.tier_records(events, trop, [], [])
    assert [e["tier"] for e in out] == [mc.CORROBORATED, mc.CONFLICT, mc.CODE_ONLY]
    assert mc.tier_summary(out) == {"myocardial_infarction": {"corroborated": 1, "conflict": 1}, "stroke": {"code_only": 1}}


def test_event_date():
    assert mc.event_date("2025-01-01", 45) == date(2025, 2, 15)


def test_sql_shapes_and_injection_guard():
    sql = mc.corroboration_sql("proj.ds", ["123", "456"])
    assert set(sql) == {"troponin", "procedures", "imaging"}
    assert "FACT_LAB_TEST" in sql["troponin"] and "'123', '456'" in sql["troponin"]
    assert "PATIENT_MERGED_FLAG = 'N'" in sql["procedures"]
    with pytest.raises(ValueError):
        mc.corroboration_sql("proj.ds", ["1'); DROP TABLE x; --"])
    with pytest.raises(ValueError):
        mc.corroboration_sql("proj.ds", [])
