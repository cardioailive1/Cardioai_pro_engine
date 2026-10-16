import csv
from datetime import date, timedelta

from training.label_schema import LabelValidator, LabelWindow
from training.mayo.mace_label_audit import (apply_to_records, auto_audit, build_review_items, export_cardiologist_queue,
                                             export_ds_audit, finalize, read_ds_audits, route, triage_ds_scope)
from training.omop_adapter import OMOPConceptMapping, build_mace_records

D0 = date(2024, 1, 1)
MAP = OMOPConceptMapping(heart_rate_concept_id=100, qtc_concept_id=101, mi_condition_concept_ids={1},
                         stroke_condition_concept_ids={2}, cv_death_cause_concept_ids={9},
                         race_concept_id_map={5: "White"})


def _cohort():
    persons, meas, conds, deaths, visits = [], [], [], [], [{"visit_occurrence_id": 1, "visit_concept_name": "Inpatient Visit"},
                                                              {"visit_occurrence_id": 2, "visit_concept_name": "Outpatient Visit"}]
    def add(pid, idx_offset=0):
        persons.append({"person_id": pid, "gender_concept_id": 8507, "race_concept_id": 5, "year_of_birth": 1960})
        for cid, v in ((100, 80), (101, 420)):
            meas.append({"person_id": pid, "measurement_concept_id": cid, "value_as_number": v, "measurement_date": D0.isoformat()})
    for pid in range(1, 9):
        add(pid)
    cond = lambda pid, off, cid=1, vid=1: conds.append({"person_id": pid, "condition_concept_id": cid, "condition_start_date": (D0 + timedelta(days=off)).isoformat(), "visit_occurrence_id": vid})
    cond(1, 45); cond(1, 46)                    # 1: clear MI at day 45, two codes, inpatient
    # 2: no events
    deaths.append({"person_id": 3, "death_date": (D0 + timedelta(days=20)).isoformat(), "cause_concept_id": None})   # 3: died, no cause, labeled none
    cond(4, 31, vid=2)                          # 4: single outpatient code at day 31 (borderline + non-acute)
    cond(5, 70, cid=2)                          # 5: stroke day 70, single code
    cond(6, 100)                                # 6: MI at day 100 (late window)
    # 7, 8: no events
    return persons, meas, conds, deaths, visits


def _fill(path, decisions):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for r in rows:
        r["auditor"] = "ds1"
        r.update(decisions.get(r["record_id"], {"ds_decision": "agree"}))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def test_full_two_tier_flow(tmp_path):
    persons, meas, conds, deaths, visits = _cohort()
    recs = build_mace_records(persons, meas, conds, deaths, MAP)
    pid = {r.patient_id: r.record_id for r in recs}
    items = build_review_items(recs, conds, deaths, MAP, visits)
    aa = auto_audit(items)
    names = lambda p: {c[0] for c in aa[pid[p]]["checks"]}
    assert aa[pid["1"]]["status"] == "pass" and aa[pid["2"]]["status"] == "pass"
    assert "death_cause_missing" in names("3") and aa[pid["3"]]["status"] == "fail"
    assert {"borderline_window", "non_acute_setting", "single_code_support"} <= names("4")
    assert "single_code_support" in names("5")

    # opt-in triage always keeps events and non-pass records in the human scope
    scope = triage_ds_scope(items, aa, sample_fraction=0.0, min_sample=0)
    assert {pid[p] for p in "13456"} <= scope

    ds = tmp_path / "ds.csv"
    n = export_ds_audit(items, str(ds), aa)
    assert n == len(recs)
    # auditor disagrees on patient 5: says it was a history code, no event
    _fill(ds, {pid["5"]: {"ds_decision": "disagree", "ds_event_type": "none", "ds_days_to_event": ""}})
    audits = read_ds_audits([str(ds)])
    plan = route(items, audits, aa)
    assert plan[pid["5"]]["state"] == "adjudicate"                    # disagreement
    assert plan[pid["3"]]["state"] == "adjudicate"                    # auto-audit fail even though auditor agreed
    assert plan[pid["1"]]["state"] == "agreed"

    card, key = tmp_path / "card.csv", tmp_path / "key.csv"
    counts = export_cardiologist_queue(items, plan, str(card), str(key), min_spot=1)
    assert counts["adjudication"] == 2 and counts["spot_check"] >= 2
    header = open(card).readline()
    assert "auto" not in header and "reason" not in header and "record_id" not in header  # blinded
    rows = list(csv.DictReader(open(card, newline="", encoding="utf-8")))
    keymap = {r["blind_id"]: r for r in csv.DictReader(open(key, newline="", encoding="utf-8"))}
    for r in rows:
        rid = keymap[r["blind_id"]]["record_id"]
        if rid == pid["5"]:
            r["cardiologist_decision"] = "none"                        # confirms: no event
        elif rid == pid["3"]:
            r["cardiologist_decision"] = "event"; r["cardiologist_event_type"] = "cardiovascular_death"; r["cardiologist_days_to_event"] = "20"
        elif rid == pid["1"]:
            r["cardiologist_decision"] = "event"; r["cardiologist_event_type"] = "myocardial_infarction"; r["cardiologist_days_to_event"] = "45"
        else:
            r["cardiologist_decision"] = "none"
    with open(card, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    res = finalize(plan, str(card), str(key), min_spot_n=1)
    assert res["unresolved"] == [] and res["outcomes"][pid["3"]] == ("cardiovascular_death", 20)
    assert res["outcomes"][pid["5"]] == ("none", None)
    assert res["spot_check"]["events"]["n"] >= 1 and res["spot_check"]["non_events"]["n"] >= 1

    fixed = {r.record_id: r for r in apply_to_records(recs, res)}
    assert fixed[pid["5"]].event_type is None and fixed[pid["5"]].label_window == LabelWindow.NONE_OBSERVED
    assert fixed[pid["3"]].label_window == LabelWindow.IMMINENT_0_30      # corrected missed CV death, 20 days
    assert fixed[pid["1"]].label_window == LabelWindow.EARLY_30_90
    LabelValidator().validate_dataset(list(fixed.values()))                # corrected records go straight into the validator


def test_pending_ds_blocks_cardiologist_queue(tmp_path):
    persons, meas, conds, deaths, visits = _cohort()
    recs = build_mace_records(persons, meas, conds, deaths, MAP)
    items = build_review_items(recs, conds, deaths, MAP)
    plan = route(items, {})
    try:
        export_cardiologist_queue(items, plan, str(tmp_path / "c.csv"), str(tmp_path / "k.csv"))
        assert False
    except ValueError as e:
        assert "not yet audited" in str(e)
