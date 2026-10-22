"""
CardioAI Pro - link each ECG to the patient's most recent EARLIER echo LVEF (Mayo Workspace)
===========================================================================================
Mayo (Adam Resnick, 2026-10-19 and 2026-10-21): FACT_ECHO_TEST and FACT_ECG both carry patient and date, more than
100,000 patients have both, dates are shifted by the same amount across a patient's whole record (so the gap in days
between an echo and an ECG is real), and LVEF is in FACT_ECHO_RESULTS. Use PATIENT_CLINIC_NUMBER as the patient id
in both row sets so the same person is one person.

Rules built in here, so they cannot be forgotten later:
  * An echo is used only if it was done ON OR BEFORE the ECG date, and no more than `max_days_before` days earlier.
    An echo after the ECG would leak the outcome (a patient who has a heart attack gets an echo afterwards).
  * Where a study has several LVEF values (biplane, visual estimate...), the study value is the MEDIAN of the plausible
    ones and the count is kept. Mayo has not said whether one study can hold several values; this makes either case safe.
  * Values outside 10-80% are dropped as implausible (the same range as the echo label audit).
  * An ECG with no usable earlier echo is kept, with has_echo = False. It is NOT dropped: most patients have no echo,
    and the model must work for them.

Not decided here, and to confirm with Mayo / the clinical lead: the name of the LVEF column in FACT_ECHO_RESULTS (the
caller passes it), and the `max_days_before` window (default 365).
"""
from __future__ import annotations

from datetime import date, datetime
from statistics import median
from typing import Any, Iterable, Optional

LVEF_MIN, LVEF_MAX = 10.0, 80.0


def _d(x: Any) -> Optional[date]:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.fromisoformat(str(x)[:10]).date()


def _num(x: Any) -> Optional[float]:
    try:
        v = float(str(x).replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return v * 100 if 0 < v <= 1 else v        # 0.55 -> 55


def echo_study_lvef(echo_rows: Iterable[dict], *, lvef_column: str, study_column: Optional[str] = None) -> tuple[list[dict], dict]:
    """echo_rows: dicts with patient_id, echo_date and the LVEF column from FACT_ECHO_RESULTS (one row per value).
    -> ([{patient_id, echo_date, lvef, n_values}], stats). One entry per study (study_column, else per patient and date)."""
    groups: dict = {}
    n_rows = n_bad_date = n_not_number = n_implausible = 0
    for r in echo_rows:
        n_rows += 1
        d = _d(r.get("echo_date"))
        if d is None:
            n_bad_date += 1
            continue
        v = _num(r.get(lvef_column))
        if v is None:
            n_not_number += 1
            continue
        if not (LVEF_MIN <= v <= LVEF_MAX):
            n_implausible += 1
            continue
        key = (str(r["patient_id"]), d, str(r.get(study_column)) if study_column else None)
        groups.setdefault(key, []).append(v)
    studies = [{"patient_id": k[0], "echo_date": k[1], "lvef": float(median(v)), "n_values": len(v)} for k, v in groups.items()]
    return studies, {"result_rows": n_rows, "studies": len(studies), "dropped_no_date": n_bad_date,
                     "dropped_not_a_number": n_not_number, "dropped_implausible": n_implausible,
                     "studies_with_several_values": sum(1 for g in groups.values() if len(g) > 1)}


def link_ecg_to_echo(ecg_rows: Iterable[dict], studies: list[dict], *, max_days_before: int = 365) -> tuple[list[dict], dict]:
    """ecg_rows: dicts with patient_id and ecg_date (plus anything else). Returns copies with an `echo` dict:
    {has_echo, lvef, days_since_echo, n_values}, and link statistics."""
    by_patient: dict = {}
    for s in studies:
        by_patient.setdefault(s["patient_id"], []).append(s)
    for v in by_patient.values():
        v.sort(key=lambda s: s["echo_date"])
    out, linked, only_after, no_date = [], 0, 0, 0
    for r in ecg_rows:
        e = _d(r.get("ecg_date"))
        best = None
        if e is None:
            no_date += 1
        else:
            cands = by_patient.get(str(r["patient_id"]), [])
            before = [s for s in cands if 0 <= (e - s["echo_date"]).days <= max_days_before]
            best = before[-1] if before else None            # latest echo that is not after the ECG
            if best is None and any(s["echo_date"] > e for s in cands):
                only_after += 1
        row = dict(r)
        if best is None:
            row["echo"] = {"has_echo": False, "lvef": None, "days_since_echo": None, "n_values": 0}
        else:
            linked += 1
            row["echo"] = {"has_echo": True, "lvef": best["lvef"], "days_since_echo": (e - best["echo_date"]).days, "n_values": best["n_values"]}
        out.append(row)
    n = len(out)
    return out, {"ecgs": n, "with_echo": linked, "share_with_echo": round(linked / n, 4) if n else None,
                 "no_ecg_date": no_date, "echo_only_after_the_ecg_not_used": only_after, "max_days_before": max_days_before}
