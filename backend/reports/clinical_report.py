"""
CardioAI Pro — Clinical Report Formatting
=============================================
The pipeline step that was completely missing: everything upstream of this
(quality, inference, diagnosis, automation tier) produces structured data
a machine can act on, but nothing turned that into something a clinician
reads. This module is that translation — one structured report per
pipeline run, suitable for rendering in the EHR, printing, or handing to a
cardiologist.

This is formatting, not a new source of clinical judgment: every fact in
the report already exists elsewhere in the pipeline's output. The report
generator's only job is assembling it into a stable, readable shape.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ClinicalReport:
    report_id: str
    patient_id: str
    modality: str
    generated_at: float
    summary: str
    vitals: dict[str, Any]
    quality: dict[str, Any]
    prediction: Optional[dict[str, Any]]
    diagnostic_finding: Optional[dict[str, Any]]
    automation: Optional[dict[str, Any]]
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "vitals": self.vitals,
            "quality": self.quality,
            "prediction": self.prediction,
            "diagnostic_finding": self.diagnostic_finding,
            "automation": self.automation,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        """Plain-text rendering — what would print or paste into an EHR note."""
        lines = [
            f"CLINICAL REPORT  {self.report_id}",
            f"Patient: {self.patient_id}    Modality: {self.modality}    Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.generated_at))}",
            "-" * 60,
            self.summary,
            "",
        ]
        if self.vitals:
            vitals_line = "  ".join(f"{k.upper()}: {v}" for k, v in self.vitals.items() if v is not None)
            if vitals_line:
                lines += ["VITALS", vitals_line, ""]
        if self.diagnostic_finding:
            lines += ["FINDING", self.diagnostic_finding.get("diagnosis", "n/a")]
            codes = self.diagnostic_finding.get("icd10_codes") or []
            if codes:
                lines.append(f"ICD-10: {', '.join(codes)}")
            lines.append("")
        if self.automation:
            lines += ["AUTOMATION TIER", f"{self.automation.get('tier', 'n/a').upper()} — {self.automation.get('action_label', '')}", ""]
        if self.recommendations:
            lines.append("RECOMMENDATIONS")
            lines += [f"  - {r}" for r in self.recommendations]
            lines.append("")
        lines.append(f"Data quality score: {self.quality.get('score', 'n/a')}/100" if self.quality else "")
        return "\n".join(l for l in lines if l is not None)


class ClinicalReportGenerator:
    def generate(
        self, patient_id: str, modality: str, vitals: dict[str, Any], quality: Optional[dict[str, Any]],
        prediction: Optional[dict[str, Any]], diagnostic_finding: Optional[dict[str, Any]],
        automation: Optional[dict[str, Any]],
    ) -> ClinicalReport:
        if diagnostic_finding and diagnostic_finding.get("specialist_review_required"):
            summary = f"{diagnostic_finding['diagnosis']}. " + (
                f"Automation tier: {automation['tier']} — {automation['action_label']}."
                if automation else "No automation decision recorded."
            )
        elif prediction:
            summary = f"No significant finding. Risk score {prediction.get('score', 'n/a')} ({prediction.get('risk_tier', 'n/a')})."
        else:
            summary = "No prediction available for this record."

        recommendations = list(automation.get("recommendations", [])) if automation else []

        return ClinicalReport(
            report_id=f"RPT-{uuid.uuid4().hex[:8]}",
            patient_id=patient_id,
            modality=modality,
            generated_at=time.time(),
            summary=summary,
            vitals=vitals or {},
            quality=quality or {},
            prediction=prediction,
            diagnostic_finding=diagnostic_finding,
            automation=automation,
            recommendations=recommendations,
        )
