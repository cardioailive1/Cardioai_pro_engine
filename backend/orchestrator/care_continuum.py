"""
CardioAI Pro — Care Continuum Pipeline
==========================================
Every other agent in this engine reasons about a single record in
isolation — one ECG reading, scored and gone. The care continuum tracker
is what ties those isolated hops into a longitudinal view of where a given
PATIENT actually is in their care journey, which is what a real command
center or diagnostics/treatment view needs: not "here's a reading," but
"here's where this patient stands right now."

Stages (linear, monotonic per patient — a patient moves forward, this
tracker never silently moves them backward):

  SCREENING           data is flowing in, nothing abnormal seen yet
  DIAGNOSTIC_REVIEW   DiagnosticAgent flagged an abnormal finding
  CARE_DECISION       AutomationTierAgent assigned an escalation tier
  TREATMENT           the finding has been actioned (mark_treatment) —
                       wires to a clinician's "mark reviewed" action
  MONITORING          back to routine surveillance after treatment
  RESOLVED            closed out — always a manual call, never automatic

This is in-memory and keyed by patient_id, same caveat as the rest of the
engine's state: swap for real persistence (Postgres, per the Render
blueprint's commented-out database block) before this needs to survive a
restart or run across multiple instances.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CareStage(Enum):
    SCREENING = "screening"
    DIAGNOSTIC_REVIEW = "diagnostic_review"
    CARE_DECISION = "care_decision"
    TREATMENT = "treatment"
    MONITORING = "monitoring"
    RESOLVED = "resolved"


STAGE_ORDER = [
    CareStage.SCREENING, CareStage.DIAGNOSTIC_REVIEW, CareStage.CARE_DECISION,
    CareStage.TREATMENT, CareStage.MONITORING, CareStage.RESOLVED,
]


@dataclass
class PatientContinuum:
    patient_id: str
    stage: CareStage = CareStage.SCREENING
    history: list[dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def advance_to(self, stage: CareStage, reason: str) -> bool:
        """Moves forward only. Returns False (no-op) if `stage` isn't strictly ahead of current."""
        if STAGE_ORDER.index(stage) <= STAGE_ORDER.index(self.stage):
            return False
        self.history.append({"from": self.stage.value, "to": stage.value, "reason": reason, "timestamp": time.time()})
        self.stage = stage
        self.last_updated = time.time()
        return True


class CareContinuumTracker:
    def __init__(self):
        self.patients: dict[str, PatientContinuum] = {}

    def _get_or_create(self, patient_id: str) -> PatientContinuum:
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientContinuum(patient_id=patient_id)
        return self.patients[patient_id]

    def record_screening(self, patient_id: str) -> None:
        self._get_or_create(patient_id)  # stays at SCREENING unless something below advances it

    def record_diagnostic_finding(self, patient_id: str, diagnosis: str) -> None:
        p = self._get_or_create(patient_id)
        if diagnosis and diagnosis != "No significant finding":
            p.advance_to(CareStage.DIAGNOSTIC_REVIEW, reason=f"finding: {diagnosis}")

    def record_care_decision(self, patient_id: str, tier: str) -> None:
        p = self._get_or_create(patient_id)
        if tier in ("urgent", "recommendation"):
            p.advance_to(CareStage.CARE_DECISION, reason=f"automation tier: {tier}")

    def mark_treatment(self, patient_id: str, note: str = "") -> bool:
        return self._get_or_create(patient_id).advance_to(CareStage.TREATMENT, reason=note or "treatment actioned")

    def mark_monitoring(self, patient_id: str) -> bool:
        return self._get_or_create(patient_id).advance_to(CareStage.MONITORING, reason="returned to routine surveillance")

    def mark_resolved(self, patient_id: str, note: str = "") -> bool:
        return self._get_or_create(patient_id).advance_to(CareStage.RESOLVED, reason=note or "manually resolved")

    def funnel_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in STAGE_ORDER}
        for p in self.patients.values():
            counts[p.stage.value] += 1
        return counts

    def status(self) -> dict[str, Any]:
        return {
            "funnel": self.funnel_counts(),
            "total_patients": len(self.patients),
            "patients": [
                {"patient_id": p.patient_id, "stage": p.stage.value, "last_updated": p.last_updated, "history_len": len(p.history)}
                for p in sorted(self.patients.values(), key=lambda x: x.last_updated, reverse=True)
            ][:50],
        }

    def patient_detail(self, patient_id: str) -> Optional[dict[str, Any]]:
        p = self.patients.get(patient_id)
        if not p:
            return None
        return {"patient_id": p.patient_id, "stage": p.stage.value, "history": p.history}
