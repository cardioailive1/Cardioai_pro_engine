"""
CardioAI Pro - ECG-level MACE labels for the 30-90 day task (Mayo Workspace)
===========================================================================
Rules fixed by the Workspace audits of 9 Nov 2026 (counts only, no patient data left the Workspace):

  * EVENTS.  MI = descendants of OMOP concept 312327 (acute MI); stroke = descendants of 443454 (cerebral infarction) and
    4310996 (ischemic stroke).  Raw condition_occurrence is NOT used: the codes are repeated on follow-up visits (about 10 rows
    per patient).  An event is the first inpatient / ER code (visit_concept_id 9201, 9203, 262) with no code of the same set in
    the prior 365 days.  Death = FACT_MORTALITY_STATUS.CONSENSUS_DEATH_DATE where CONSENSUS_STATUS = 'deceased'.  The OMOP `death`
    table holds only about 10% of those deaths (25,808 vs 253,055), so it is not used.
  * INDEX.  ECG date = DATE(ACQUISITION_TS) of the resting Rhythm waveform.  Event dates are DATEs, so windows count whole days.
  * CLASSES (checked in this order): no person match; contradicted death status (patient excluded); ECG after the follow-up cutoff;
    a MACE episode in the prior 90 days (washout); first event 0-29 days (imminent, excluded - see label_schema); first event
    30-89 days = POSITIVE; otherwise NEGATIVE only if the patient is shown alive and in the system 90 days after the ECG (a later
    visit, or status 'verifiably_alive'); anything else has no follow-up evidence and is excluded (unknown outcome, not a negative).
  * CUTOFF.  FOD_DATA_CUTOFF_DATE (27 Mar 2026) is when the external death source stops, so the last usable ECG date is that
    minus the 90-day horizon (27 Dec 2025).  Later ECGs cannot be labeled.
  * Mayo records no cause of death, so ~73% of positives are all-cause deaths.  Report the composite AND the non-fatal MI/stroke
    subset (positive_kind = 'condition') against the age/sex baseline.  Split by patient, and cap ECGs per patient.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

from .bq import table_ref

INPATIENT_ER_VISIT_CONCEPTS = (9201, 9203, 262)
EPISODE_GAP_DAYS = 365
WASHOUT_DAYS = 90
IMMINENT_DAYS = 30
HORIZON_DAYS = 90

NO_PERSON = "0_no_person_match"
CONTRADICTED = "1_excluded_contradicted_death_status"
AFTER_CUTOFF = "2_excluded_after_followup_cutoff"
PRIOR_EVENT = "3_excluded_prior_event_within_90d"
IMMINENT = "4_excluded_imminent_0_30d"
NO_FOLLOW_UP = "5_excluded_no_follow_up_evidence"
NEGATIVE = "6_negative_confirmed_alive_90d"
POSITIVE = "7_POSITIVE_event_30_90d"


def last_usable_ecg_date(fod_cutoff: date, horizon_days: int = HORIZON_DAYS) -> date:
    """Last ECG date whose full follow-up window ends on or before the death-data cutoff."""
    return fod_cutoff - timedelta(days=horizon_days)


def classify(*, has_person: bool, status: Optional[str], ecg_date: date, ecg_cutoff: date, prior_gap: Optional[int],
             first_fwd: Optional[int], last_visit: Optional[date]) -> str:
    """Pure-Python twin of the CASE in `mace_label_sql`; `prior_gap` = days since the latest earlier episode, `first_fwd` = days to
    the first event on or after the ECG date (either may be None)."""
    if not has_person:
        return NO_PERSON
    if status == "contradicted":
        return CONTRADICTED
    if ecg_date > ecg_cutoff:
        return AFTER_CUTOFF
    if prior_gap is not None and prior_gap <= WASHOUT_DAYS:
        return PRIOR_EVENT
    if first_fwd is not None and first_fwd < IMMINENT_DAYS:
        return IMMINENT
    if first_fwd is not None and first_fwd < HORIZON_DAYS:
        return POSITIVE
    if (last_visit is not None and last_visit >= ecg_date + timedelta(days=HORIZON_DAYS)) or status == "verifiably_alive":
        return NEGATIVE
    return NO_FOLLOW_UP


def mace_label_sql(schema: str, concept_ids: Iterable[int], ecg_cutoff: date, summary: bool = False) -> str:
    """BigQuery SQL labelling every resting Rhythm ECG.  summary=True -> counts per class (safe to print);
    summary=False -> one row per ECG (pcn, ts, class, positive_kind, days_to_event): patient-level, keep inside the Workspace."""
    ids = ", ".join(str(int(i)) for i in concept_ids)
    if not ids:
        raise ValueError("concept_ids is empty")
    visits = ", ".join(str(int(v)) for v in INPATIENT_ER_VISIT_CONCEPTS)
    T = lambda t: table_ref(schema, t)
    final = (
        "SELECT class, IF(class LIKE '7%', first_kind, NULL) AS positive_kind, COUNT(*) AS ecgs, COUNT(DISTINCT pcn) AS patients\n"
        "FROM c GROUP BY 1, 2 ORDER BY 1, 2"
        if summary else
        "SELECT pcn, ts, class, IF(class LIKE '7%', first_kind, NULL) AS positive_kind, first_fwd AS days_to_event\nFROM c"
    )
    return f"""
WITH cond AS (
  SELECT DISTINCT co.person_id, co.condition_start_date AS d
  FROM {T("condition_occurrence")} co
  JOIN {T("visit_occurrence")} v ON v.visit_occurrence_id = co.visit_occurrence_id
  WHERE co.condition_concept_id IN ({ids}) AND v.visit_concept_id IN ({visits})
),
epi AS (
  SELECT person_id, d FROM (
    SELECT person_id, d, LAG(d) OVER (PARTITION BY person_id ORDER BY d) AS prev FROM cond)
  WHERE prev IS NULL OR DATE_DIFF(d, prev, DAY) > {EPISODE_GAP_DAYS}
),
mort AS (
  SELECT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS pcn_s, CONSENSUS_STATUS AS st, CONSENSUS_DEATH_DATE AS dd
  FROM {T("FACT_MORTALITY_STATUS")}
),
ev AS (
  SELECT person_id, d, 'condition' AS kind FROM epi
  UNION ALL
  SELECT p.person_id, m.dd, 'death' FROM mort m
  JOIN {T("person")} p ON p.person_source_value = m.pcn_s
  WHERE m.st = 'deceased' AND m.dd IS NOT NULL
),
lastv AS (SELECT person_id, MAX(visit_start_date) AS last_visit FROM {T("visit_occurrence")} GROUP BY 1),
acq AS (
  SELECT DISTINCT PATIENT_CLINIC_NUMBER AS pcn, ACQUISITION_TS AS ts
  FROM {T("FACT_ECG_WAVEFORM")}
  WHERE LOWER(WAVEFORM_TYPE) = 'rhythm' AND LOWER(EKG_TYPE) = 'resting'
),
a2 AS (
  SELECT a.pcn, a.ts, DATE(a.ts) AS d0, p.person_id, mo.st, lv.last_visit
  FROM acq a
  LEFT JOIN {T("person")} p ON p.person_source_value = CAST(a.pcn AS STRING)
  LEFT JOIN mort mo ON mo.pcn_s = CAST(a.pcn AS STRING)
  LEFT JOIN lastv lv ON lv.person_id = p.person_id
),
a3 AS (
  SELECT a2.pcn, a2.ts, ANY_VALUE(a2.d0) AS d0, ANY_VALUE(a2.person_id) AS person_id,
    ANY_VALUE(a2.st) AS st, ANY_VALUE(a2.last_visit) AS last_visit,
    MIN(IF(e.d >= a2.d0, DATE_DIFF(e.d, a2.d0, DAY), NULL)) AS first_fwd,
    MIN(IF(e.d < a2.d0 AND e.kind = 'condition', DATE_DIFF(a2.d0, e.d, DAY), NULL)) AS prior_gap,
    ARRAY_AGG(IF(e.d >= a2.d0, e.kind, NULL) IGNORE NULLS ORDER BY e.d LIMIT 1)[SAFE_OFFSET(0)] AS first_kind
  FROM a2 LEFT JOIN ev e ON e.person_id = a2.person_id
  GROUP BY a2.pcn, a2.ts
),
c AS (
  SELECT *,
    CASE WHEN person_id IS NULL THEN '{NO_PERSON}'
         WHEN st = 'contradicted' THEN '{CONTRADICTED}'
         WHEN d0 > DATE '{ecg_cutoff.isoformat()}' THEN '{AFTER_CUTOFF}'
         WHEN prior_gap <= {WASHOUT_DAYS} THEN '{PRIOR_EVENT}'
         WHEN first_fwd < {IMMINENT_DAYS} THEN '{IMMINENT}'
         WHEN first_fwd < {HORIZON_DAYS} THEN '{POSITIVE}'
         WHEN last_visit >= DATE_ADD(d0, INTERVAL {HORIZON_DAYS} DAY) OR st = 'verifiably_alive' THEN '{NEGATIVE}'
         ELSE '{NO_FOLLOW_UP}' END AS class
  FROM a3
)
{final}
""".strip()
