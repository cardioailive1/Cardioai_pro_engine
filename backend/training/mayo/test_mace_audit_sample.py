from datetime import date, datetime

import pytest

from training.mayo.mace_audit_sample import (DEFAULT_PER_STRATUM, audit_mapping, audit_sample_sql, build_audit_records, evidence_sql)
from training.mayo.mace_label_audit import auto_audit, build_review_items
from training.mayo.mace_label_rules import NEGATIVE, POSITIVE

MI, STROKE = [312327], [443454]
D0 = date(2024, 3, 1)


def row(pcn, person, cls, days=None, kind=None, prior=None, d0=D0):
    return {"pcn": pcn, "ts": datetime(2024, 3, 1, 9, 30, 5), "d0": d0, "person_id": person, "class": cls,
            "positive_kind": kind, "days_to_event": days, "prior_gap": prior, "stratum": "x"}


def cond(person, concept, offset, visit):
    from datetime import timedelta
    return {"person_id": person, "condition_concept_id": concept, "condition_concept_name": "x",
            "condition_start_date": D0 + timedelta(days=offset), "visit_occurrence_id": visit}


def test_sample_sql_is_stratified_one_ecg_per_patient_and_reproducible():
    sql = audit_sample_sql("p.d", MI + STROKE, date(2025, 12, 27), seed="s1")
    for part in ("PARTITION BY stratum, pcn", "PARTITION BY stratum ORDER BY FARM_FINGERPRINT", "'s1'", "rn = 1", "rk <=",
                 "'pos_condition'", "'pos_death'", "'negative'", "DATE '2025-12-27'", "9201, 9203, 262"):
        assert part in sql
    assert str(DEFAULT_PER_STRATUM["negative"]) in sql
    assert sql.count("WITH ") == 1                        # spliced onto the label query, not nested
    assert audit_sample_sql("p.d", MI, date(2025, 12, 27), seed="s1") == audit_sample_sql("p.d", MI, date(2025, 12, 27), seed="s1")
    with pytest.raises(ValueError):
        audit_sample_sql("p.d", MI, date(2025, 12, 27), seed="x'; DROP")


def test_evidence_sql_refuses_unsafe_ids():
    q = evidence_sql("p.d", MI, [1, 2], ["A1", "B-2"])
    assert set(q) == {"conditions", "visits", "deaths"} and "'A1', 'B-2'" in q["deaths"]
    with pytest.raises(ValueError):
        evidence_sql("p.d", MI, [1], ["A1') OR 1=1 --"])
    with pytest.raises(ValueError):
        evidence_sql("p.d", MI, [], ["A1"])


def test_records_feed_the_real_audit_end_to_end():
    rows = [row("P1", 1, POSITIVE, 45, "condition", prior=400),
            row("P2", 2, POSITIVE, 60, "death"),
            row("P3", 3, NEGATIVE),
            row("P4", 4, NEGATIVE)]
    conds = [cond(1, 312327, 45, 11),                      # inpatient acute MI, one code only
             cond(4, 312327, 120, 44),                     # outpatient MI code after the 90 days: negative, code present
             cond(4, 312327, -30, 43)]
    deaths = [{"person_id": "P2", "death_date": date(2024, 4, 30), "cause_concept_id": 0}]
    recs, ev, dth = build_audit_records(rows, conds, deaths, MI, STROKE)
    assert [r.event_type.value if r.event_type else None for r in recs] == ["myocardial_infarction", "all_cause_death", None, None]
    assert recs[2].follow_up_days == 90 and recs[2].censored and recs[0].days_since_prior_mace == 400
    assert all(c["person_id"].startswith("P") for c in ev)             # evidence re-keyed to PATIENT_CLINIC_NUMBER
    items = build_review_items(recs, ev, dth, audit_mapping(MI, STROKE),
                               [{"visit_occurrence_id": 11, "visit_concept_name": "Inpatient Visit"},
                                {"visit_occurrence_id": 44, "visit_concept_name": "Outpatient Visit"},
                                {"visit_occurrence_id": 43, "visit_concept_name": "Outpatient Visit"}])
    aa = auto_audit(items)
    assert set(aa) == {r.record_id for r in recs}
    names = {rid: {c[0] for c in a["checks"]} for rid, a in aa.items()}
    r1, r3, r4 = (recs[0].record_id, recs[2].record_id, recs[3].record_id)
    assert "single_code_support" in names[r1]
    assert aa[r3]["status"] == "pass"                                   # clean negative
    assert "code_present_but_no_event" in names[r4] and aa[r4]["status"] == "warn"


def test_positive_condition_without_a_code_on_the_event_day_raises():
    with pytest.raises(ValueError, match="no MI/stroke code on day 45"):
        build_audit_records([row("P1", 1, POSITIVE, 45, "condition")], [cond(1, 312327, 50, 1)], [], MI, STROKE)


def test_stroke_vs_mi_and_non_sample_classes():
    recs, _, _ = build_audit_records([row("P1", 1, POSITIVE, 40, "condition")], [cond(1, 443454, 40, 1)], [], MI, STROKE)
    assert recs[0].event_type.value == "stroke"
    with pytest.raises(ValueError, match="not part of the audit sample"):
        build_audit_records([row("P1", 1, "4_excluded_imminent_0_30d")], [], [], MI, STROKE)


def test_type_comes_from_inpatient_er_codes_only_when_visits_are_given():
    rows = [row("P1", 1, POSITIVE, 85, "condition")]
    conds = [cond(1, 312327, 85, 10), cond(1, 443454, 85, 11)]       # outpatient MI + inpatient stroke on the event day
    visits = [{"visit_occurrence_id": 10, "visit_concept_id": 9202}, {"visit_occurrence_id": 11, "visit_concept_id": 9201}]
    recs, _, _ = build_audit_records(rows, conds, [], MI, STROKE, visits)
    assert recs[0].event_type.value == "stroke"
    recs, _, _ = build_audit_records(rows, conds, [], MI, STROKE)      # old behaviour without visit rows
    assert recs[0].event_type.value == "myocardial_infarction"
