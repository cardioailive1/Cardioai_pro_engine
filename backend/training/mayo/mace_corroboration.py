"""
CardioAI Pro - independent corroboration of the MACE event labels (Mayo Workspace)
==================================================================================
The automatic audit checks that the rule was applied consistently; it reads the same diagnosis codes the rule used, so it cannot
tell whether a code was a real event.  This module adds evidence from OTHER tables, per sampled event, and assigns a tier:

  MI        corroborated  an elevated troponin within +-TROP_DAYS of the event day, or a PCI / CABG within +-PROC_DAYS
            conflict      troponin was measured around the event and every interpretable result was normal, no PCI / CABG
            code_only     no interpretable troponin and no revascularization around the event
  stroke    corroborated  brain / carotid imaging within +-IMG_DAYS of the event day
            code_only     none
  death     corroborated  consensus mortality source (not a diagnosis code), so no further check

`corroborated` records can skip human review; `conflict` and `code_only` go to the data scientists.  The cardiologist sample
stays: it is what measures accuracy against clinical judgment.  Everything here holds patient-level data: keep it in the Workspace.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date, timedelta
from typing import Iterable, Optional

from training.omop_adapter import _to_date

from .bq import table_ref

TROP_DAYS, PROC_DAYS, IMG_DAYS = 3, 7, 2
CORROBORATED, CONFLICT, CODE_ONLY = "corroborated", "conflict", "code_only"

PCI_CABG_REGEX = r"^(9292[0-9]|9293[0-9]|9294[0-4]|3351[0-9]|3352[0-9]|3353[0-6])$"
IMAGING_REGEX = r"brain|head|cerebr|carotid|stroke|intracran"
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def _quoted(values: Iterable[str], what: str) -> str:
    vals = [str(v) for v in values]
    bad = [v for v in vals if not _SAFE.match(v)]
    if bad:
        raise ValueError(f"{what}: refusing values that are not plain ids (e.g. {bad[:2]!r})")
    if not vals:
        raise ValueError(f"{what} is empty")
    return ", ".join(f"'{v}'" for v in vals)


# ---------------------------------------------------------------- troponin parsing (pure)
def parse_result(result_txt) -> Optional[tuple]:
    """'46' -> ('=', 46.0); '<0.02' -> ('<', 0.02); '>50' -> ('>', 50.0); CANCELED / blank / text -> None."""
    s = str(result_txt or "").strip()
    m = re.match(r"^([<>]=?)?\s*(" + _NUM.pattern + r")\s*[A-Za-z/%]*$", s)
    if not m:
        return None
    return (m.group(1) or "=", float(m.group(2)))


def parse_upper_limit(normal_range_txt) -> Optional[float]:
    """'<=15' -> 15; '0.00 - 0.12' -> 0.12; '<0.04' -> 0.04; blank -> None."""
    nums = _NUM.findall(str(normal_range_txt or ""))
    return float(nums[-1]) if nums else None


def troponin_elevated(result_txt, normal_range_txt) -> Optional[bool]:
    """True / False when the result can be read against its own reference range; None when it cannot."""
    parsed, upper = parse_result(result_txt), parse_upper_limit(normal_range_txt)
    if parsed is None or upper is None:
        return None
    op, val = parsed
    if op.startswith("<"):
        return False if val <= upper else None
    if op.startswith(">"):
        return True if val >= upper else None
    return val > upper


# ---------------------------------------------------------------- tiers (pure)
def _near(d, event: date, days: int) -> bool:
    return abs((_to_date(d) - event).days) <= days


def event_date(d0, days_to_event) -> date:
    return _to_date(d0) + timedelta(days=int(days_to_event))


def corroborate(event_type: str, event: date, troponin: list, procedures: list, imaging: list) -> tuple:
    """-> (tier, reason).  troponin rows: (date, result_txt, normal_range_txt); procedures / imaging rows: (date, ...)."""
    event = _to_date(event)
    if event_type == "all_cause_death":
        return CORROBORATED, "consensus mortality source"
    if event_type == "myocardial_infarction":
        if any(_near(r[0], event, PROC_DAYS) for r in procedures):
            return CORROBORATED, "PCI/CABG within %d days" % PROC_DAYS
        reads = [troponin_elevated(r[1], r[2]) for r in troponin if _near(r[0], event, TROP_DAYS)]
        if any(v is True for v in reads):
            return CORROBORATED, "elevated troponin within %d days" % TROP_DAYS
        if any(v is False for v in reads):
            return CONFLICT, "troponin measured around the event, all normal"
        return CODE_ONLY, "no interpretable troponin or PCI/CABG"
    if event_type == "stroke":
        if any(_near(r[0], event, IMG_DAYS) for r in imaging):
            return CORROBORATED, "brain/carotid imaging within %d days" % IMG_DAYS
        return CODE_ONLY, "no brain/carotid imaging"
    raise ValueError(f"unknown event_type {event_type!r}")


def tier_records(events: list, troponin_rows: list, procedure_rows: list, imaging_rows: list) -> list:
    """events: dicts with key, pcn, event_type, event_date.  Rows are dicts keyed by pcn (see the *_sql columns).
    -> events with 'tier' and 'tier_reason' added."""
    def by_pcn(rows, *fields):
        out: dict = {}
        for r in rows:
            out.setdefault(str(r["pcn"]), []).append(tuple(r[f] for f in fields))
        return out
    trop, proc, img = by_pcn(troponin_rows, "d", "result_txt", "normal_range_txt"), by_pcn(procedure_rows, "d"), by_pcn(imaging_rows, "d")
    out = []
    for e in events:
        pcn = str(e["pcn"])
        tier, why = corroborate(e["event_type"], e["event_date"], trop.get(pcn, []), proc.get(pcn, []), img.get(pcn, []))
        out.append({**e, "tier": tier, "tier_reason": why})
    return out


def tier_summary(tiered: list) -> dict:
    """{event_type: {tier: count}} for reporting."""
    out: dict = {}
    for e in tiered:
        out.setdefault(e["event_type"], Counter())[e["tier"]] += 1
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------- SQL
def corroboration_sql(schema: str, pcns: Iterable[str]) -> dict:
    """{'troponin', 'procedures', 'imaging'}: rows for just the sampled patients, each with pcn and d (a DATE)."""
    T = lambda t: table_ref(schema, t)
    pc = _quoted(pcns, "pcns")
    patient = f"JOIN {T('DIM_PATIENT')} p ON p.PATIENT_DK = f.PATIENT_DK AND p.PATIENT_MERGED_FLAG = 'N'"
    return {
        "troponin": f"""
SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS pcn, DATE(f.LAB_COLLECTION_DTM) AS d, f.RESULT_TXT AS result_txt,
       f.NORMAL_RANGE_TXT AS normal_range_txt
FROM {T("FACT_LAB_TEST")} f
JOIN {T("DIM_LAB_TEST")} t ON t.LAB_TEST_DK = f.LAB_TEST_DK
{patient}
WHERE REGEXP_CONTAINS(LOWER(t.LAB_TEST_DESCRIPTION), r'troponin') AND f.RESULT_TXT IS NOT NULL
  AND UPPER(f.RESULT_TXT) NOT LIKE '%CANCEL%' AND CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ({pc})""".strip(),
        "procedures": f"""
SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS pcn, DATE(f.PROCEDURE_DTM) AS d, c.PROCEDURE_CODE AS code
FROM {T("FACT_PROCEDURES")} f
JOIN {T("DIM_PROCEDURE_CODE")} c ON c.PROCEDURE_CODE_DK = f.PROCEDURE_CODE_DK
{patient}
WHERE REGEXP_CONTAINS(c.PROCEDURE_CODE, r'{PCI_CABG_REGEX}') AND CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ({pc})""".strip(),
        "imaging": f"""
SELECT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS pcn, DATE(RECORDED_DATE) AS d, STUDY_DESCRIPTION AS description
FROM {T("FACT_IMAGINGSTUDY")}
WHERE REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(STUDY_DESCRIPTION, ''), ' ', IFNULL(PROCEDURE_CODE_TEXT, ''))), r'{IMAGING_REGEX}')
  AND CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ({pc})""".strip(),
    }
