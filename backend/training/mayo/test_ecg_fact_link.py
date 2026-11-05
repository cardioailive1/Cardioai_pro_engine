from datetime import datetime
from training.mayo.ecg_fact_link import (collapse_fact_ecg, group_waveform_acquisitions, link_waveform_to_fact_ecg,
                                         ecg_waveform_sql, fact_ecg_sql)

T0 = "2026-01-05 11:45:00"


def _fact(seq, text, **kw):
    return {"PATIENT_DK": 7, "ECG_DTM": T0, "ENCOUNTER_NUMBER": "E1", "ECG_INTERPRET_SEQ_NBR": seq,
            "ECG_INTERPRET_TEXT": text, "ECG_INTERPRET_FULL_TEXT": "Normal ECG", **kw}


def test_several_fact_rows_collapse_to_one_ecg_in_sequence_order():
    c = collapse_fact_ecg([_fact(2, "Normal axis"), _fact(1, "Normal sinus rhythm"), _fact(3, "Normal ECG")])
    assert len(c) == 1
    e = next(iter(c.values()))
    assert e["interpretation_snippets"] == ["Normal sinus rhythm", "Normal axis", "Normal ECG"] and e["source_rows"] == 3
    assert e["ENCOUNTER_NUMBER"] == "E1" and not e["encounter_conflict"]


def test_waveform_rows_group_by_patient_and_time_and_keep_rhythm_only():
    rows = [{"PATIENT_DK": 7, "PATIENT_CLINIC_NUMBER": "C7", "ACQUISITION_TS": T0, "LEAD_ID": l, "WAVEFORM_TYPE": "rhythm", "WAVEFORM": "x"}
            for l in ("I", "II")]
    rows.append({"PATIENT_DK": 7, "ACQUISITION_TS": T0, "LEAD_ID": "I", "WAVEFORM_TYPE": "median", "WAVEFORM": "y"})
    g = group_waveform_acquisitions(rows)
    assert len(g) == 1 and len(g[0]["records"]) == 2


def test_link_uses_the_ecg_time_and_never_a_filing_time():
    acq = group_waveform_acquisitions([{"PATIENT_DK": 7, "ACQUISITION_TS": T0, "LEAD_ID": "I", "WAVEFORM_TYPE": "rhythm", "WAVEFORM": "x"}])
    facts = collapse_fact_ecg([_fact(1, "a", DATA_AVAILABLE_DTM="2026-01-05 13:15:00")])
    r = link_waveform_to_fact_ecg(acq, facts)
    assert len(r["linked"]) == 1 and r["linked"][0]["index_time"] == datetime(2026, 1, 5, 11, 45)
    assert "DATA_AVAILABLE_DTM" not in fact_ecg_sql("p.d")


def test_unmatched_and_ambiguous_are_reported_not_guessed():
    acq = group_waveform_acquisitions([{"PATIENT_DK": 7, "ACQUISITION_TS": T0, "LEAD_ID": "I", "WAVEFORM_TYPE": "rhythm", "WAVEFORM": "x"}])
    assert link_waveform_to_fact_ecg(acq, {})["unmatched"]
    facts = collapse_fact_ecg([_fact(1, "a"), {**_fact(1, "b"), "ECG_DTM": "2026-01-05 11:45:20"}])
    r = link_waveform_to_fact_ecg(acq, facts, tolerance_seconds=30)
    assert r["ambiguous"] and not r["linked"]
    assert link_waveform_to_fact_ecg(acq, facts)["linked"]   # exact match by default


def test_sql_joins_dim_patient_and_filters_resting_rhythm():
    s = ecg_waveform_sql("p.d")
    assert "DIM_PATIENT" in s and "'rhythm'" in s and "'resting'" in s and "PATIENT_MERGED_FLAG = 'N'" in s


def test_sample_rate_and_heart_rate_are_carried_through():
    acq = group_waveform_acquisitions([{"PATIENT_DK": 7, "ACQUISITION_TS": T0, "LEAD_ID": "I", "WAVEFORM_TYPE": "rhythm",
                                        "WAVEFORM": "x", "ECGSAMPLEBASE": 500}])
    assert acq[0]["sample_rate_hz"] == 500.0
    e = next(iter(collapse_fact_ecg([_fact(1, "a", HEART_RATE=72)]).values()))
    assert e["heart_rate_bpm"] == 72.0
    assert "FACT_ECG_WAVEFORM_RESTING_MEASUREMENT" in ecg_waveform_sql("p.d") and "e.HEART_RATE" in fact_ecg_sql("p.d")
