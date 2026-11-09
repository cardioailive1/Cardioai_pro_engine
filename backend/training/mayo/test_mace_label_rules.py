from datetime import date

import pytest

from training.mayo.mace_label_rules import (AFTER_CUTOFF, CONTRADICTED, IMMINENT, NEGATIVE, NO_FOLLOW_UP, NO_PERSON, POSITIVE,
                                            PRIOR_EVENT, classify, last_usable_ecg_date, mace_label_sql)

D0, CUT = date(2024, 1, 10), date(2025, 12, 27)
base = dict(has_person=True, status="unknown", ecg_date=D0, ecg_cutoff=CUT, prior_gap=None, first_fwd=None, last_visit=None)


def c(**kw):
    return classify(**{**base, **kw})


def test_cutoff_is_death_data_cutoff_minus_horizon():
    assert last_usable_ecg_date(date(2026, 3, 27)) == date(2025, 12, 27)


def test_window_boundaries():
    assert c(first_fwd=29) == IMMINENT
    assert c(first_fwd=30) == POSITIVE
    assert c(first_fwd=89) == POSITIVE
    assert c(first_fwd=90, status="verifiably_alive") == NEGATIVE       # 90 days out is not in the 30-90 window
    assert c(first_fwd=0) == IMMINENT


def test_washout_boundary_and_order():
    assert c(prior_gap=90, first_fwd=45) == PRIOR_EVENT
    assert c(prior_gap=91, first_fwd=45) == POSITIVE
    assert c(prior_gap=10, first_fwd=5) == PRIOR_EVENT                  # washout is checked before imminent


def test_exclusions_and_censoring():
    assert c(has_person=False) == NO_PERSON
    assert c(status="contradicted", first_fwd=45) == CONTRADICTED
    assert c(ecg_date=date(2025, 12, 28), first_fwd=45) == AFTER_CUTOFF
    assert c(ecg_date=CUT, first_fwd=45) == POSITIVE
    assert c() == NO_FOLLOW_UP                                          # no event and no sign they were still in the system
    assert c(last_visit=date(2024, 4, 8)) == NO_FOLLOW_UP               # 89 days later: not enough
    assert c(last_visit=date(2024, 4, 9)) == NEGATIVE                   # exactly 90 days later
    assert c(status="verifiably_alive") == NEGATIVE


def test_positive_needs_no_follow_up_evidence():
    assert c(first_fwd=60) == POSITIVE


def test_sql_carries_the_rules():
    sql = mace_label_sql("p.d", [312327, 443454, 4310996], CUT)
    for part in ("312327, 443454, 4310996", "9201, 9203, 262", "> 365", "DATE '2025-12-27'", "prior_gap <= 90",
                 "first_fwd < 30", "first_fwd < 90", "CONSENSUS_STATUS", "'deceased'", "FACT_ECG_WAVEFORM"):
        assert part in sql
    assert "`death`" not in sql and ".death`" not in sql                # the OMOP death table is deliberately unused
    assert "COUNT(*) AS ecgs" in mace_label_sql("p.d", [1], CUT, summary=True)
    assert "days_to_event" in sql


def test_sql_refuses_bad_input():
    with pytest.raises(ValueError):
        mace_label_sql("p.d", [], CUT)
    with pytest.raises(ValueError):
        mace_label_sql("p.d", ["1; DROP TABLE x"], CUT)
