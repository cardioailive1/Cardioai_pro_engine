"""
CardioAI Pro - linking radiology reports to DICOM images (Mayo Clinic Platform)
==============================================================================
From Mayo's Radiology training slides (2026-10-15):
  FACT_RADIOLOGY.ACCESSION_NBR  <->  FACT_RADIOLOGY_DICOM_INVENTORY.DICOM_ACCESSION_NUMBER
  Inventory -> DICOM headers (FACT_RADIOLOGY_DICOM_HEADER_STORAGE): join on ALL of
  STMO_NAME_IN_ADL, MODALITY, STUDY_INSTANCE_UID, SERIES_INSTANCE_UID, SOP_INSTANCE_UID
  Inventory -> FACT_IMAGINGSTUDY (curated study header): ALL of
  STMO_NAME_IN_ADL, MODALITY, STUDY_INSTANCE_UID
  Only IMAGE_DEID_STATUS in ('Ready','Available') can be submitted for de-identification.
  DICOM inventory is NOT mapped into OMOP; reports are also in OMOP note.note_text.

Not covered by the slides: the echo-note (FACT_ECHO_TEST) link to images - still
to confirm with Mayo. Echo DICOM may sit in the same inventory (ultrasound modality);
check STUDY_DESCRIPTION / FACT_IMAGINGSTUDY.PROCEDURE_CODE_TEXT in the Workspace.
Table/schema qualifiers are left to the caller; column names are as shown in the slides.
"""
from __future__ import annotations

from typing import Iterable

SUBMITTABLE = {"ready", "available"}


def cardiac_mri_study_sql(schema: str = "") -> str:
    """Spark SQL: cardiac MRI studies (curated header) joined to reports and inventory.
    Refine the LIKE terms with PROCEDURE_CODE_TEXT values you actually see in FACT_IMAGINGSTUDY."""
    q = (schema + ".") if schema else ""
    return f"""
SELECT s.STUDY_INSTANCE_UID, s.PROCEDURE_CODE_TEXT, s.STUDY_DESCRIPTION, s.NUMBER_OF_SERIES, s.NUMBER_OF_INSTANCES,
       i.DICOM_ACCESSION_NUMBER, i.IMAGE_DEID_STATUS, i.SERIES_INSTANCE_UID, i.SERIES_DESCRIPTION,
       r.RADIOLOGY_REPORT, r.RADIOLOGY_IMPRESSION, r.RADIOLOGY_DTM
FROM {q}FACT_IMAGINGSTUDY s
JOIN {q}FACT_RADIOLOGY_DICOM_INVENTORY i
  ON  i.STMO_NAME_IN_ADL = s.STMO_NAME_IN_ADL
  AND i.MODALITY = s.MODALITY_1
  AND i.STUDY_INSTANCE_UID = s.STUDY_INSTANCE_UID
JOIN {q}FACT_RADIOLOGY r ON r.ACCESSION_NBR = i.DICOM_ACCESSION_NUMBER
WHERE s.MODALITY_1 = 'MR'
  AND (UPPER(s.PROCEDURE_CODE_TEXT) LIKE '%CARDIAC%' OR UPPER(s.STUDY_DESCRIPTION) LIKE '%CARDIAC%'
       OR UPPER(s.STUDY_DESCRIPTION) LIKE '%HEART%')
  AND UPPER(i.IMAGE_DEID_STATUS) IN ('READY','AVAILABLE')
""".strip()


def echo_dicom_study_sql(schema: str = "") -> str:
    """Spark SQL: echocardiogram DICOM studies in the inventory. Mayo (Adam Resnick, 2026-10-28): echo is listed in
    FACT_RADIOLOGY_DICOM_INVENTORY with DICOM modality 'US'; find echo by searching STUDY_DESCRIPTION. Exactly which
    descriptions are echo (TTE, TEE, stress, contrast) is for a person to confirm from the distinct values - this lists them
    so they can be counted before anything is requested. Confirm the column names in the Workspace."""
    q = (schema + ".") if schema else ""
    return f"""
SELECT i.STUDY_DESCRIPTION, COUNT(DISTINCT i.STUDY_INSTANCE_UID) AS studies
FROM {q}FACT_RADIOLOGY_DICOM_INVENTORY i
WHERE i.MODALITY = 'US' AND UPPER(i.STUDY_DESCRIPTION) LIKE '%ECHO%'
  AND UPPER(i.IMAGE_DEID_STATUS) IN ('READY','AVAILABLE')
GROUP BY i.STUDY_DESCRIPTION
ORDER BY studies DESC
""".strip()


def link_reports_to_images(reports: Iterable[dict], inventory: Iterable[dict]) -> dict:
    """reports: rows with ACCESSION_NBR (+ RADIOLOGY_REPORT). inventory: rows with
    DICOM_ACCESSION_NUMBER, STUDY_INSTANCE_UID, SERIES_INSTANCE_UID, SOP_INSTANCE_UID, IMAGE_DEID_STATUS.
    Returns {'linked': {accession: {'report': row, 'studies': {uid: [instances]}}}, 'no_images': [...],
    'not_submittable': [...]}. Accession numbers match exactly (no fuzzy matching)."""
    by_acc: dict = {}
    for r in inventory:
        acc = r.get("DICOM_ACCESSION_NUMBER")
        if acc not in (None, ""):
            by_acc.setdefault(str(acc), []).append(r)
    linked, no_images, not_sub = {}, [], []
    for rep in reports:
        acc = rep.get("ACCESSION_NBR")
        rows = by_acc.get(str(acc)) if acc not in (None, "") else None
        if not rows:
            no_images.append(acc); continue
        ok = [x for x in rows if str(x.get("IMAGE_DEID_STATUS", "")).strip().lower() in SUBMITTABLE]
        if not ok:
            not_sub.append(acc); continue
        studies: dict = {}
        for x in ok:
            studies.setdefault(x["STUDY_INSTANCE_UID"], []).append(
                {"series": x.get("SERIES_INSTANCE_UID"), "sop": x.get("SOP_INSTANCE_UID")})
        linked[str(acc)] = {"report": rep, "studies": studies}
    return {"linked": linked, "no_images": no_images, "not_submittable": not_sub}


DEID_COLUMNS = ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]


def write_deid_request_csv(rows: Iterable[dict], path: str) -> dict:
    """Mayo (TSM) de-identification request: one row per selected imaging STUDY with exactly
    STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL (that order). `rows` come from the
    Workspace (inventory / FACT_IMAGINGSTUDY joined to the patient key). Rows with a missing
    value, or whose IMAGE_DEID_STATUS (if present) is not Ready/Available, are rejected and
    reported, never silently written. Duplicates (same study) collapse to one row."""
    import csv
    seen, kept, rejected = set(), [], []
    for r in rows:
        st = r.get("IMAGE_DEID_STATUS")
        if any(str(r.get(c, "")).strip() == "" for c in DEID_COLUMNS):
            rejected.append((r.get("STUDY_INSTANCE_UID"), "missing value")); continue
        if st is not None and str(st).strip().lower() not in SUBMITTABLE:
            rejected.append((r.get("STUDY_INSTANCE_UID"), f"IMAGE_DEID_STATUS={st}")); continue
        key = str(r["STUDY_INSTANCE_UID"]).strip()
        if key in seen:
            continue
        seen.add(key); kept.append({c: str(r[c]).strip() for c in DEID_COLUMNS})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DEID_COLUMNS); w.writeheader(); w.writerows(kept)
    return {"written": len(kept), "rejected": rejected}
