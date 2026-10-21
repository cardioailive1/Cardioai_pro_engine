"""
OMOP -> payer-style training rows. One row per (patient, as_of date): features from the 12 months BEFORE
as_of using only claims-like inputs (claims_risk.features), label = first MI / stroke / cardiovascular
death in the 12 months AFTER. Reuses OMOPConceptMapping for the outcome concept IDs.

Payers have enrollment, OMOP has OBSERVATION_PERIOD: a row is built only if the patient is observed for the
whole lookback and follow-up window (otherwise a 'no event' could just mean the patient left). Patients who
died before as_of are skipped. Diagnosis codes come from condition_source_value (ICD-10 as recorded; dots
are stripped). Visit concept IDs default to the standard OMOP inpatient (9201) and emergency room (9203)
visits; confirm against your instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Iterable, Optional

from claims_risk.features import vectorize
from training.omop_adapter import OMOPConceptMapping, _subgroups, _to_date


@dataclass
class ClaimsRiskRecord:
    patient_id: str
    as_of: str
    x: list[float]
    label: int
    subgroups: dict[str, str] = field(default_factory=dict)


def build_claims_risk_records(person_rows, condition_rows, visit_rows, observation_rows, death_rows,
                              mapping: OMOPConceptMapping, as_of_dates: Iterable, lookback_days: int = 365,
                              horizon_days: int = 365, inpatient_concept_ids=frozenset({9201}),
                              ed_concept_ids=frozenset({9203})) -> list[ClaimsRiskRecord]:
    conds, visits, obs, deaths = {}, {}, {}, {}
    for c in condition_rows:
        conds.setdefault(c["person_id"], []).append(c)
    for v in visit_rows or []:
        visits.setdefault(v["person_id"], []).append(v)
    for o in observation_rows:
        obs.setdefault(o["person_id"], []).append(o)
    for d in death_rows or []:
        deaths.setdefault(d["person_id"], []).append(d)
    outcome_ids = mapping.mi_condition_concept_ids | mapping.stroke_condition_concept_ids
    out = []
    for p in person_rows:
        pid = p["person_id"]
        yob = p.get("year_of_birth")
        for as_of_raw in as_of_dates:
            as_of = _to_date(as_of_raw)
            lo, hi = as_of - timedelta(days=lookback_days), as_of + timedelta(days=horizon_days)
            covered = any(_to_date(o["observation_period_start_date"]) <= lo and _to_date(o["observation_period_end_date"]) >= hi
                          for o in obs.get(pid, []))
            died_before = any(_to_date(d.get("death_date")) is not None and _to_date(d["death_date"]) < as_of for d in deaths.get(pid, []))
            if not covered or died_before or yob is None:
                continue
            codes = [c.get("condition_source_value") or "" for c in conds.get(pid, [])
                     if lo <= _to_date(c["condition_start_date"]) < as_of]
            vs = [v for v in visits.get(pid, []) if lo <= _to_date(v["visit_start_date"]) < as_of]
            n_ip = sum(1 for v in vs if v["visit_concept_id"] in inpatient_concept_ids)
            n_ed = sum(1 for v in vs if v["visit_concept_id"] in ed_concept_ids)
            gender = p.get("gender_concept_id")
            male = True if gender == mapping.male_gender_concept_id else False if gender == mapping.female_gender_concept_id else None
            event = any(c["condition_concept_id"] in outcome_ids and as_of < _to_date(c["condition_start_date"]) <= hi for c in conds.get(pid, []))
            event = event or any(_to_date(d.get("death_date")) is not None and as_of < _to_date(d["death_date"]) <= hi
                                 and mapping.death_counts(d) for d in deaths.get(pid, []))
            out.append(ClaimsRiskRecord(str(pid), as_of.isoformat(), vectorize(as_of.year - int(yob), male, codes, n_ip if visit_rows is not None else None, n_ed if visit_rows is not None else None),
                                        int(event), _subgroups(p, mapping, as_of)))
    return out
