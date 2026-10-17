"""
Claims-style features for the payer risk model. Built ONLY from what a payer claims feed carries
(age, sex, ICD-10 diagnosis codes, inpatient / ED visit counts), so a model trained on Mayo's OMOP
history can score payer members with the same inputs. Diagnosis groups use ICD-10 code families
(not OMOP concept IDs), so they mean the same thing on both sides.

The ICD-10 groups below are a clinical choice made by CardioAI, NOT a validated risk-adjustment model
(not CMS-HCC). The clinical team should review and edit DX_GROUPS before real use.
"""
from __future__ import annotations

from typing import Iterable

DX_GROUPS: dict[str, tuple[str, ...]] = {
    "hypertension": ("I10", "I11", "I12", "I13", "I15", "I16"),
    "diabetes": ("E10", "E11", "E13"),
    "ckd": ("N18",),
    "heart_failure": ("I50", "I110", "I130", "I132"),
    "atrial_fibrillation": ("I48",),
    "coronary_disease": ("I20", "I24", "I25"),
    "prior_mi": ("I21", "I22", "I252"),
    "prior_stroke_tia": ("I63", "I64", "G45", "Z8673"),
    "peripheral_artery_disease": ("I70", "I73"),
    "lipid_disorder": ("E78",),
    "copd": ("J44",),
    "obesity": ("E66",),
}
FEATURE_NAMES = ["age", "male", "sex_known"] + list(DX_GROUPS) + ["n_distinct_dx", "inpatient_visits", "ed_visits", "visits_known"]


def normalize_code(code: str) -> str:
    return (code or "").replace(".", "").strip().upper()


def vectorize(age: float, male, dx_codes: Iterable[str], inpatient_visits=None, ed_visits=None) -> list[float]:
    """male: True/False/None (unknown). Visit counts None = not available in this feed."""
    codes = {normalize_code(c) for c in dx_codes if c}
    row = [float(age), 1.0 if male else 0.0, 0.0 if male is None else 1.0]
    row += [1.0 if any(c.startswith(p) for c in codes for p in prefixes) else 0.0 for prefixes in DX_GROUPS.values()]
    row += [float(min(len(codes), 30)), float(min(inpatient_visits or 0, 10)), float(min(ed_visits or 0, 10)),
            0.0 if inpatient_visits is None and ed_visits is None else 1.0]
    return row
