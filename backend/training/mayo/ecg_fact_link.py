"""
CardioAI Pro - link ECG waveforms to the clinical ECG record (FACT_ECG)
======================================================================
From Mayo's ECG Data slides (3 Nov 2026):
  * FACT_ECG_WAVEFORM.ECG_WAVEFORM_FPK does NOT join to FACT_ECG.ECG_FPK.
  * Link: waveform PATIENT_CLINIC_NUMBER -> DIM_PATIENT -> PATIENT_DK, then
    FACT_ECG_WAVEFORM.ACQUISITION_TS = FACT_ECG.ECG_DTM.  ENCOUNTER_NUMBER is available after reaching FACT_ECG.
  * One ECG has several FACT_ECG rows (interpretation snippets, ordered by ECG_INTERPRET_SEQ_NBR); study-level
    values repeat across them, so the rows are collapsed to ONE ECG before anything is counted.
  * Use the physiological time (ACQUISITION_TS / ECG_DTM). DATA_AVAILABLE_DTM and ECG_DOCUMENT_UPDATE_DTM are later
    filing times and are never used as the ECG time (outcome windows start at the ECG time).
  * Mayo (Tommy Stretch, 5 Nov): every interpretation on the platform was reviewed and signed by a cardiologist in MUSE, and the
    resting measurement (ECGSAMPLEBASE) joins the waveform on the patient and ACQUISITION_TS. In the Workspace dataset
    (verified 9 Nov 2026) FACT_ECG_WAVEFORM_RESTING_MEASUREMENT has no PATIENT_DK; it joins on PATIENT_CLINIC_NUMBER + ACQUISITION_TS. Heart rate: use FACT_ECG.HEART_RATE
    (the final clinician-validated value).
  * The interpretation is machine-proposed, then confirmed, edited and signed by the cardiologist in MUSE.
Column names are as shown in the slides; confirm them in the Workspace. The SQL runs through the BigQuery client (bq.py).
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from .bq import table_ref

LATER_FILING_COLUMNS = ("DATA_AVAILABLE_DTM", "ECG_DOCUMENT_UPDATE_DTM")


def ecg_waveform_sql(schema: str = "") -> str:
    """Resting rhythm waveform rows (one row per lead, decoded on its own), with PATIENT_DK from DIM_PATIENT."""
    T = lambda t: table_ref(schema, t)
    return f"""
SELECT w.ECG_WAVEFORM_FPK, w.PATIENT_CLINIC_NUMBER, p.PATIENT_DK, w.ACQUISITION_TS, w.LEAD_ID, w.WAVEFORM_TYPE, w.WAVEFORM,
       m.ECGSAMPLEBASE, m.ECGSAMPLEEXPONENT
FROM {T("FACT_ECG_WAVEFORM")} w
JOIN {T("DIM_PATIENT")} p ON p.PATIENT_CLINIC_NUMBER = w.PATIENT_CLINIC_NUMBER AND p.PATIENT_MERGED_FLAG = 'N'
LEFT JOIN {T("FACT_ECG_WAVEFORM_RESTING_MEASUREMENT")} m ON m.PATIENT_CLINIC_NUMBER = w.PATIENT_CLINIC_NUMBER AND m.ACQUISITION_TS = w.ACQUISITION_TS
WHERE LOWER(w.WAVEFORM_TYPE) = 'rhythm' AND LOWER(w.EKG_TYPE) = 'resting'
""".strip()


def fact_ecg_sql(schema: str = "") -> str:
    """FACT_ECG interpretation rows (several per ECG). Filing times are deliberately not selected."""
    T = lambda t: table_ref(schema, t)
    return f"""
SELECT e.PATIENT_DK, e.ECG_DTM, e.ENCOUNTER_NUMBER, e.HEART_RATE, e.ECG_INTERPRET_SEQ_NBR, e.ECG_INTERPRET_TEXT,
       e.ECG_INTERPRET_FULL_TEXT, e.ECG_INTERPRET_CODE_DK, e.ECG_INTERPRET_EXTENSION_TEXT
FROM {T("FACT_ECG")} e
""".strip()


def _ts(x) -> Optional[datetime]:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)
    s = str(x).strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unreadable timestamp: {x!r}")


def group_waveform_acquisitions(rows: Iterable[dict]) -> list:
    """Waveform rows -> one entry per (patient, ACQUISITION_TS): {PATIENT_DK, PATIENT_CLINIC_NUMBER, acquisition_ts, records}.
    `records` go straight to ecg_waveform.assemble_mayo_ecg. Only resting rhythm rows are kept."""
    out = {}
    for r in rows:
        if str(r.get("WAVEFORM_TYPE", "")).lower() != "rhythm":
            continue
        ts = _ts(r.get("ACQUISITION_TS"))
        if ts is None:
            continue
        key = (str(r.get("PATIENT_DK")), ts)
        g = out.setdefault(key, {"PATIENT_DK": str(r.get("PATIENT_DK")), "PATIENT_CLINIC_NUMBER": r.get("PATIENT_CLINIC_NUMBER"),
                                 "acquisition_ts": ts, "records": [], "sample_rate_hz": None})
        if r.get("ECGSAMPLEBASE") not in (None, ""):
            g["sample_rate_hz"] = float(r["ECGSAMPLEBASE"])
        g["records"].append({"LEAD_ID": r["LEAD_ID"], "WAVEFORM_TYPE": r["WAVEFORM_TYPE"], "WAVEFORM": r["WAVEFORM"]})
    return list(out.values())


def collapse_fact_ecg(rows: Iterable[dict]) -> dict:
    """FACT_ECG rows -> {(PATIENT_DK, ECG_DTM): one ECG}. Interpretation snippets are ordered by ECG_INTERPRET_SEQ_NBR and
    joined; repeated study-level values are taken once. `encounter_conflict` is True when the rows disagree on the encounter."""
    groups = {}
    for r in rows:
        ts = _ts(r.get("ECG_DTM"))
        if ts is None:
            continue
        groups.setdefault((str(r.get("PATIENT_DK")), ts), []).append(r)
    out = {}
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: (r.get("ECG_INTERPRET_SEQ_NBR") is None, r.get("ECG_INTERPRET_SEQ_NBR") or 0))
        snippets = [str(r["ECG_INTERPRET_TEXT"]).strip() for r in rs if r.get("ECG_INTERPRET_TEXT")]
        full = next((r["ECG_INTERPRET_FULL_TEXT"] for r in rs if r.get("ECG_INTERPRET_FULL_TEXT")), None)
        enc = {r.get("ENCOUNTER_NUMBER") for r in rs if r.get("ENCOUNTER_NUMBER") not in (None, "")}
        codes = sorted({str(r["ECG_INTERPRET_CODE_DK"]) for r in rs if r.get("ECG_INTERPRET_CODE_DK") not in (None, "")})
        out[key] = {"PATIENT_DK": key[0], "ecg_time": key[1], "ENCOUNTER_NUMBER": next(iter(enc)) if len(enc) == 1 else None,
                    "encounter_conflict": len(enc) > 1,
                    "heart_rate_bpm": next((float(r["HEART_RATE"]) for r in rs if r.get("HEART_RATE") not in (None, "")), None), "interpretation_snippets": snippets, "interpretation_full_text": full,
                    "interpretation_code_dks": codes,
                    "extension_text": next((r["ECG_INTERPRET_EXTENSION_TEXT"] for r in rs if r.get("ECG_INTERPRET_EXTENSION_TEXT")), None),
                    "source_rows": len(rs)}
    return out


def link_waveform_to_fact_ecg(acquisitions: Iterable[dict], fact_ecgs: dict, tolerance_seconds: float = 0.0) -> dict:
    """Match each waveform acquisition to a collapsed FACT_ECG on PATIENT_DK and ACQUISITION_TS = ECG_DTM (exact by default).
    -> {"linked": [...], "unmatched": [...], "ambiguous": [...]}. A linked entry carries `index_time` (the ECG time that
    outcome windows start from), the waveform `records` and the clinical `ecg`. More than one FACT_ECG within the tolerance
    is ambiguous and is not guessed."""
    by_patient = {}
    for (dk, ts), e in fact_ecgs.items():
        by_patient.setdefault(dk, []).append((ts, e))
    linked, unmatched, ambiguous = [], [], []
    for a in acquisitions:
        cands = [e for ts, e in by_patient.get(str(a["PATIENT_DK"]), [])
                 if abs((ts - a["acquisition_ts"]).total_seconds()) <= tolerance_seconds]
        if not cands:
            unmatched.append(a)
        elif len(cands) > 1:
            ambiguous.append({**a, "candidates": len(cands)})
        else:
            linked.append({**a, "index_time": a["acquisition_ts"], "ecg": cands[0]})
    return {"linked": linked, "unmatched": unmatched, "ambiguous": ambiguous}
