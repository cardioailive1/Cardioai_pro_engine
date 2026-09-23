"""
CardioAI Pro — Diagnostic Engine
====================================
Bridges risk scoring (InferenceAgent) to an actual diagnosis + ICD-10
coding that AutomationTierAgent can act on and the care continuum tracker
can log. Without this, "automation tier" had nothing real feeding it —
it was evaluating a hardcoded placeholder string.

IMPORTANT — deliberately conservative by design, not by omission:
With only summary ECG features (heart rate, QTc, HRV) and no rhythm
morphology, imaging, or lab data, this engine cannot honestly assign
disease-specific diagnoses like STEMI, atrial fibrillation, or heart
failure — a cardiologist distinguishes those from ST-segment morphology,
echo findings, and troponin trends, none of which this pipeline ingests.

So when risk is elevated, this engine assigns only a GENERIC abnormal-
finding code (R94.31 — abnormal results of cardiovascular function
studies) and explicitly flags the case for specialist correlation, rather
than guessing a specific disease. This is a safety property, not a
limitation to work around: a system that confidently prints "STEMI" from
three summary numbers would be actively dangerous. Disease-specific ICD-10
coding happens downstream — by a clinician or coder (see the clinician
dashboard's billing view) — or by a real, validated diagnostic model wired
in via load_trained_diagnostic_model() below, once one exists with the
morphology/imaging/lab data to back it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

GENERIC_ABNORMAL_CODE = "R94.31"
GENERIC_ABNORMAL_DESC = "Abnormal results of cardiovascular function studies"


def load_trained_diagnostic_model() -> Optional[Callable]:
    """
    Hook for a real, validated diagnostic classifier — one trained on ECG
    morphology, echo, and labs, not just summary features. Returning None
    (the default) keeps this engine on the conservative generic-finding
    path described in the module docstring.
    """
    return None


@dataclass
class DiagnosticFinding:
    diagnosis: str
    icd10_codes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    specialist_review_required: bool = False
    basis: list[str] = field(default_factory=list)


class DiagnosticEngine:
    def __init__(self):
        self._trained_model = load_trained_diagnostic_model()

    def evaluate(self, record: dict[str, Any], quality_score: float, prediction: Optional[dict[str, Any]]) -> DiagnosticFinding:
        if self._trained_model is not None:
            return self._trained_model(record, quality_score, prediction)

        if not prediction:
            return DiagnosticFinding(diagnosis="No significant finding", confidence=round(quality_score, 2))

        risk_tier = prediction.get("risk_tier", "low")
        basis = list(prediction.get("inputs_used", []))

        if risk_tier == "high":
            return DiagnosticFinding(
                diagnosis="Abnormal ECG pattern, high MACE risk — specialist correlation required",
                icd10_codes=[GENERIC_ABNORMAL_CODE],
                confidence=round(quality_score, 2),
                specialist_review_required=True,
                basis=basis,
            )
        if risk_tier == "moderate":
            return DiagnosticFinding(
                diagnosis="Borderline ECG findings",
                icd10_codes=[GENERIC_ABNORMAL_CODE],
                confidence=round(quality_score * 0.85, 2),
                specialist_review_required=True,
                basis=basis,
            )
        return DiagnosticFinding(diagnosis="No significant finding", confidence=round(quality_score, 2), basis=basis)
