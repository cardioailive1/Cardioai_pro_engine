"""
CardioAI Pro — FHIR R4 Integration
======================================
Builds spec-shaped FHIR R4 resources (Patient, Observation, RiskAssessment,
DiagnosticReport) from normalized records. These are the resources CardioScan
Pro would POST back into Epic/Cerner via their FHIR API, or expose for a
hospital's EHR to pull.

Coding systems used:
  - LOINC   for Observation.code (vital signs, ECG measures)
  - SNOMED CT for RiskAssessment / condition coding
  - CardioAI's own CodeSystem for the proprietary MACE risk score

This module only *shapes* data to spec — it does not persist to a FHIR
server. Wire `FHIRBuilder.push()` to a real FHIR server's REST API
(e.g. Epic's `POST /Observation`) when a hospital connection is live.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

LOINC = {
    "heart_rate": ("8867-4", "Heart rate"),
    "spo2": ("59408-5", "Oxygen saturation"),
    "qt_interval": ("8634-8", "QTc interval"),
}


def _fhir_id() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class FHIRBuilder:
    def from_record(self, modality: str, record: dict[str, Any], record_id: str) -> list[dict]:
        if modality in ("ecg", "wearable"):
            return self._vitals_bundle(modality, record, record_id)
        if modality == "claims":
            return [self._risk_assessment_stub(record, record_id)]
        if modality == "imaging":
            return [self._diagnostic_report_stub(record, record_id)]
        return []

    def _observation(self, subject_ref: str, code_key: str, value: float, unit: str) -> dict:
        code, display = LOINC[code_key]
        return {
            "resourceType": "Observation",
            "id": _fhir_id(),
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                       "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
            "subject": {"reference": subject_ref},
            "effectiveDateTime": _now(),
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"},
        }

    def _vitals_bundle(self, modality: str, record: dict[str, Any], record_id: str) -> list[dict]:
        subject_id = record.get("patient_id") or record.get("subscriber_id") or record_id
        subject_ref = f"Patient/{subject_id}"
        obs = []
        if "heart_rate_bpm" in record and record["heart_rate_bpm"] is not None:
            obs.append(self._observation(subject_ref, "heart_rate", record["heart_rate_bpm"], "/min"))
        if "spo2_pct" in record and record.get("spo2_pct") is not None:
            obs.append(self._observation(subject_ref, "spo2", record["spo2_pct"], "%"))
        if "qt_interval_ms" in record and record.get("qt_interval_ms") is not None:
            obs.append(self._observation(subject_ref, "qt_interval", record["qt_interval_ms"], "ms"))
        return obs

    def _risk_assessment_stub(self, record: dict[str, Any], record_id: str) -> dict:
        member_id = record.get("member_id", record_id)
        return {
            "resourceType": "RiskAssessment",
            "id": _fhir_id(),
            "status": "final",
            "subject": {"reference": f"Patient/{member_id}"},
            "occurrenceDateTime": _now(),
            "basis": [{"reference": "Coverage/claims-derived"}],
            "note": [{"text": "Derived from de-identified payer claims pool (CardioAI population model)."}],
        }

    def _diagnostic_report_stub(self, record: dict[str, Any], record_id: str) -> dict:
        study_id = record.get("study_instance_uid", record_id)
        return {
            "resourceType": "DiagnosticReport",
            "id": _fhir_id(),
            "status": "preliminary",
            "code": {"text": "Cardiac imaging AI-assisted read"},
            "imagingStudy": [{"reference": f"ImagingStudy/{study_id}"}],
            "effectiveDateTime": _now(),
        }

    def push(self, resource: dict, base_url: str) -> None:
        """
        Placeholder for a real FHIR server push, e.g.:
            requests.post(f"{base_url}/{resource['resourceType']}", json=resource,
                           headers={"Authorization": f"Bearer {token}"})
        Left unimplemented here — wire to a hospital's Epic/Cerner FHIR
        endpoint and OAuth2 (SMART on FHIR) credentials once connected.
        """
        raise NotImplementedError("Connect to a live FHIR server to enable push().")
