"""
CardioAI Pro - audit sample and label records for the MACE outcome-label review (Mayo Workspace)
===============================================================================================
Step between the automatic labels (`mace_label_rules.mace_label_sql`, 5.9M ECGs) and the review in `mace_label_audit`:

  1. `audit_sample_sql`   draws a reproducible stratified sample INSIDE BigQuery: one ECG per patient, per stratum
                          (positive MI/stroke, positive death, confirmed negative).  Same rules as the full labels.
  2. `evidence_sql`       the MI/stroke codes (with visit setting) and consensus deaths for just those patients.
  3. `build_audit_records` MACELabelRecords + evidence rows in the shape `build_review_items` expects.  The patient key is
                          PATIENT_CLINIC_NUMBER everywhere (the audit evidence is re-keyed to it), so the patient-level split
                          and the audit use the same id.
  4. then `build_review_items` -> `auto_audit` (REQUIRED) -> `export_ds_audit` -> `route` -> `export_cardiologist_queue`
     -> `finalize`   (see RUNBOOK 'Reviewing the MACE outcome labels').

What the review certifies: it is a SAMPLE audit.  The spot-check agreement (target >= 95%) measures how often the automatic rule
agrees with the data scientists and the cardiologist, per stratum, with a Wilson interval.  Corrections apply to the sampled
records; the other ECGs keep their automatic label.  Systematic errors the reviewers find (e.g. a code that is really history)
are fixed in the RULE and the labels regenerated, then re-audited.  Everything here holds patient-level data: keep it inside the
Workspace.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from training.label_schema import MACEEventType, MACELabelRecord
from training.omop_adapter import OMOPConceptMapping, _to_date

from .bq import table_ref
from .mace_label_rules import HORIZON_DAYS, INPATIENT_ER_VISIT_CONCEPTS, NEGATIVE, POSITIVE, mace_label_sql

DEFAULT_PER_STRATUM = {"pos_condition": 200, "pos_death": 200, "negative": 300}
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")

# The row-level SELECT that `mace_label_sql(summary=False)` ends with; replaced here by the sampling tail.
_ROW_SELECT = "SELECT pcn, ts, class, IF(class LIKE '7%', first_kind, NULL) AS positive_kind, first_fwd AS days_to_event\nFROM c"


def _ints(values: Iterable[int], what: str) -> str:
    out = ", ".join(str(int(v)) for v in values)
    if not out:
        raise ValueError(f"{what} is empty")
    return out


def _quoted(values: Iterable[str], what: str) -> str:
    vals = [str(v) for v in values]
    bad = [v for v in vals if not _SAFE.match(v)]
    if bad:
        raise ValueError(f"{what}: refusing values that are not plain ids (e.g. {bad[:2]!r})")
    if not vals:
        raise ValueError(f"{what} is empty")
    return ", ".join(f"'{v}'" for v in vals)


def audit_sample_sql(schema: str, concept_ids: Iterable[int], ecg_cutoff, per_stratum: Optional[dict] = None, seed: str = "audit1") -> str:
    """One row per sampled ECG: pcn, ts, d0, person_id, class, positive_kind, days_to_event, prior_gap, stratum.
    One ECG per patient per stratum, ranked by a hash of (patient, seed) so the draw is reproducible."""
    n = {**DEFAULT_PER_STRATUM, **(per_stratum or {})}
    if not _SAFE.match(seed):
        raise ValueError("seed must be letters, digits, _ or -")
    base = mace_label_sql(schema, concept_ids, ecg_cutoff, summary=False)
    if _ROW_SELECT not in base:
        raise RuntimeError("mace_label_sql changed: update _ROW_SELECT in mace_audit_sample.py")
    tail = f"""
, s AS (
  SELECT *, CASE WHEN class = '{POSITIVE}' AND first_kind = 'condition' THEN 'pos_condition'
                 WHEN class = '{POSITIVE}' THEN 'pos_death'
                 WHEN class = '{NEGATIVE}' THEN 'negative' END AS stratum
  FROM c
),
one AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY stratum, pcn ORDER BY FARM_FINGERPRINT(CONCAT(CAST(ts AS STRING), '{seed}'))) AS rn
  FROM s WHERE stratum IS NOT NULL
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY stratum ORDER BY FARM_FINGERPRINT(CONCAT(pcn, '{seed}'))) AS rk
  FROM one WHERE rn = 1
)
SELECT pcn, ts, d0, person_id, class, IF(class LIKE '7%', first_kind, NULL) AS positive_kind, first_fwd AS days_to_event,
       prior_gap, stratum
FROM ranked
WHERE rk <= CASE stratum WHEN 'pos_condition' THEN {int(n['pos_condition'])} WHEN 'pos_death' THEN {int(n['pos_death'])}
                         ELSE {int(n['negative'])} END
ORDER BY stratum, rk
""".strip("\n")
    return base.replace(_ROW_SELECT, tail)


def evidence_sql(schema: str, concept_ids: Iterable[int], person_ids: Iterable[int], pcns: Iterable[str]) -> dict:
    """SQL for the evidence of the sampled patients only: {'conditions', 'visits', 'deaths'}.  Condition rows are ALL
    qualifying MI/stroke codes (any setting), so reviewers can see re-coded history the labeling rule ignores."""
    T = lambda t: table_ref(schema, t)
    ids, cids, pc = _ints(person_ids, "person_ids"), _ints(concept_ids, "concept_ids"), _quoted(pcns, "pcns")
    return {
        "conditions": f"""
SELECT co.person_id, co.condition_concept_id, c.concept_name AS condition_concept_name, co.condition_start_date, co.visit_occurrence_id
FROM {T("condition_occurrence")} co JOIN {T("concept")} c ON c.concept_id = co.condition_concept_id
WHERE co.person_id IN ({ids}) AND co.condition_concept_id IN ({cids})""".strip(),
        "visits": f"""
SELECT v.visit_occurrence_id, v.visit_concept_id, c.concept_name AS visit_concept_name
FROM {T("visit_occurrence")} v JOIN {T("concept")} c ON c.concept_id = v.visit_concept_id
WHERE v.visit_occurrence_id IN (SELECT visit_occurrence_id FROM {T("condition_occurrence")}
                                WHERE person_id IN ({ids}) AND condition_concept_id IN ({cids}))""".strip(),
        "deaths": f"""
SELECT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS person_id, CONSENSUS_DEATH_DATE AS death_date, 0 AS cause_concept_id
FROM {T("FACT_MORTALITY_STATUS")}
WHERE CONSENSUS_STATUS = 'deceased' AND CONSENSUS_DEATH_DATE IS NOT NULL AND CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ({pc})""".strip(),
    }


def audit_mapping(mi_ids: Iterable[int], stroke_ids: Iterable[int]) -> OMOPConceptMapping:
    """Mapping for the review (all-cause death).  The ECG feature concept ids are not used by the review."""
    return OMOPConceptMapping(heart_rate_concept_id=0, qtc_concept_id=0, mi_condition_concept_ids=set(map(int, mi_ids)),
                              stroke_condition_concept_ids=set(map(int, stroke_ids)), death_definition="all_cause")


def build_audit_records(sample_rows: list, condition_rows: list, death_rows: list, mi_ids: Iterable[int], stroke_ids: Iterable[int],
                        visit_rows: Optional[list] = None):
    """-> (records, condition_rows_by_pcn, death_rows) ready for `build_review_items`.
    A positive condition record takes MI or stroke from the code on the event day (MI if both); a record with no code on that day
    is a data error and raises.  Pass `visit_rows` (visit_occurrence_id, visit_concept_id from `evidence_sql`) so only inpatient / ER
    codes decide the type, as in the labeling rule; without it, an outpatient MI code on the same day as an inpatient stroke would
    wrongly make the record an MI.  Negatives carry 90 days of confirmed follow-up (LabelValidator needs >= 90)."""
    mi, stroke = set(map(int, mi_ids)), set(map(int, stroke_ids))
    ip_visits = None if visit_rows is None else {str(v["visit_occurrence_id"]) for v in visit_rows
                                                 if int(v["visit_concept_id"]) in INPATIENT_ER_VISIT_CONCEPTS}
    pcn_of = {str(r["person_id"]): str(r["pcn"]) for r in sample_rows}
    conds = [{**c, "person_id": pcn_of[str(c["person_id"])]} for c in condition_rows if str(c["person_id"]) in pcn_of]
    by_pcn: dict = {}
    for c in conds:
        by_pcn.setdefault(c["person_id"], []).append(c)
    records = []
    for r in sample_rows:
        pcn, d0 = str(r["pcn"]), _to_date(r["d0"])
        ts = re.sub(r"\D", "", str(r["ts"]))[:14]
        rid, cls, days = f"MACE-{pcn}-{ts}", r["class"], r.get("days_to_event")
        if cls == POSITIVE:
            days = int(days)
            if r.get("positive_kind") == "death":
                et = MACEEventType.ALL_CAUSE_DEATH
            else:
                on_day = {int(c["condition_concept_id"]) for c in by_pcn.get(pcn, []) if (_to_date(c["condition_start_date"]) - d0).days == days
                          and (ip_visits is None or str(c.get("visit_occurrence_id")) in ip_visits)}
                if on_day & mi:
                    et = MACEEventType.MYOCARDIAL_INFARCTION
                elif on_day & stroke:
                    et = MACEEventType.STROKE
                else:
                    raise ValueError(f"{rid}: positive condition record but no MI/stroke code on day {days}")
            rec = dict(event_type=et, days_to_event=days, follow_up_days=days, censored=False)
        elif cls == NEGATIVE:
            rec = dict(event_type=None, days_to_event=None, follow_up_days=HORIZON_DAYS, censored=True)
        else:
            raise ValueError(f"{rid}: class {cls!r} is not part of the audit sample")
        prior = r.get("prior_gap")
        records.append(MACELabelRecord(record_id=rid, patient_id=pcn, modality="ecg", index_timestamp=0.0, features={},
                                       days_since_prior_mace=int(prior) if prior is not None else None,
                                       index_date=d0.isoformat(), **rec))
    return records, conds, [dict(d, person_id=str(d["person_id"])) for d in death_rows]
