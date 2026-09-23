"""
CardioAI Pro — Automation Tier Engine
=========================================
Decides how much autonomy the engine exercises for a diagnostic finding:

  URGENT          auto-escalate now — no human gate on the alert firing
  RECOMMENDATION  surfaced to a clinician as an actionable suggestion,
                  requires sign-off before anything happens
  INFORMATIVE     logged for reference only, no action implied

Confidence here is NOT the risk/MACE score itself — it's how much to trust
that score, aggregated across whichever modalities (ECG, echo, labs)
contributed to the finding. A high-risk score computed from a single,
low-quality, incomplete record should not auto-escalate; that's exactly
what the tiering gate below prevents by requiring BOTH high confidence
AND an urgent-coded diagnosis before anything reaches URGENT — high
confidence alone, or an urgent code alone, is deliberately not enough.

Where this sits in the pipeline: this engine does not diagnose anything
itself. It takes a diagnosis + its ICD-10 coding (assigned by a clinician,
a coder, or eventually a trained diagnostic model) and decides the
escalation policy for acting on it. See orchestrator/agents.py's
AutomationTierAgent for how the live ECG-only pipeline calls this today —
deliberately unable to reach URGENT, since nothing in the real-time stream
carries a confirmed, coded diagnosis yet (see that file for why).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AutomationTier(Enum):
    URGENT = "urgent"
    RECOMMENDATION = "recommendation"
    INFORMATIVE = "informative"


# Real ICD-10 codes: STEMI involving other coronary artery of anterior/inferior
# wall. Presence of one of these AND high cross-modal confidence is the only
# path to URGENT — this list is intentionally narrow.
URGENT_ICD10_CODES = ["I21.19", "I21.09"]

CONFIDENCE_URGENT_MIN = 95.0
CONFIDENCE_RECOMMENDATION_MIN = 80.0


@dataclass
class AutomationDecision:
    tier: AutomationTier
    confidence: float
    icd10_codes: List[str]
    diagnosis: str
    recommendations: List[str]
    requires_human_signoff: bool
    auto_escalate: bool
    action_label: str = ""


class AutomationTierEngine:
    """Stateless policy engine — safe to call concurrently, no shared mutable state."""

    TIER_ACTION_LABEL: Dict[AutomationTier, str] = {
        AutomationTier.URGENT: "Auto-escalated — on-call cardiologist paged",
        AutomationTier.RECOMMENDATION: "Queued as recommendation — awaiting clinician sign-off",
        AutomationTier.INFORMATIVE: "Logged for reference only — no action taken",
    }

    # Diagnosis-specific playbooks. Falls back to GENERIC_ACTIONS for any
    # diagnosis string not listed here (see _generate_recommendations).
    DIAGNOSIS_ACTIONS: Dict[str, Dict[AutomationTier, List[str]]] = {
        "STEMI": {
            AutomationTier.URGENT: [
                "Activate STEMI protocol",
                "Page on-call interventional cardiologist immediately",
                "Prepare cath lab — door-to-balloon clock started",
                "Notify ED attending and charge nurse",
            ],
            AutomationTier.RECOMMENDATION: [
                "Recommend urgent cardiology consult",
                "Consider serial troponins and repeat ECG in 15-30 min",
                "Flag for physician review within the hour",
            ],
            AutomationTier.INFORMATIVE: [
                "Findings logged for physician review",
                "Confidence or coding did not meet the urgent-tier bar — no automated escalation",
            ],
        },
        "Atrial fibrillation": {
            AutomationTier.RECOMMENDATION: [
                "Recommend rate/rhythm control evaluation",
                "Consider anticoagulation risk assessment (e.g. CHA2DS2-VASc)",
                "Flag for cardiology follow-up within 24-48h",
            ],
            AutomationTier.INFORMATIVE: [
                "Findings logged for physician review at next visit",
                "No immediate action indicated by automation — clinical correlation advised",
            ],
        },
        "Heart failure, systolic": {
            AutomationTier.RECOMMENDATION: [
                "Recommend BNP/NT-proBNP and volume status assessment",
                "Consider diuretic titration per guideline-directed therapy",
                "Flag for cardiology follow-up within 1 week",
            ],
            AutomationTier.INFORMATIVE: [
                "Findings logged for physician review",
                "Trend against prior echo/labs at next visit",
            ],
        },
    }

    GENERIC_ACTIONS: Dict[AutomationTier, List[str]] = {
        AutomationTier.URGENT: [
            "Auto-escalated to on-call physician — high-confidence urgent finding",
            "Immediate clinical correlation required",
        ],
        AutomationTier.RECOMMENDATION: [
            "Recommend clinician review within 24 hours",
            "Consider correlating with additional diagnostics before acting",
        ],
        AutomationTier.INFORMATIVE: [
            "Findings logged for physician review",
            "No automated action taken — confidence below recommendation threshold",
        ],
    }

    def _calculate_confidence(self, ecg: Dict, echo: Optional[Dict], lab: Optional[Dict]) -> float:
        confidences = [ecg.get("confidence", 0)]
        if echo and echo.get("confidence"):
            confidences.append(echo["confidence"])
        if lab and lab.get("confidence"):
            confidences.append(lab["confidence"])
        return sum(confidences) / len(confidences) if confidences else 70.0

    def _get_automation_tier(self, confidence: float, icd10_codes: List[str]) -> AutomationTier:
        if any(code in icd10_codes for code in URGENT_ICD10_CODES) and confidence >= CONFIDENCE_URGENT_MIN:
            return AutomationTier.URGENT
        if confidence >= CONFIDENCE_RECOMMENDATION_MIN:
            return AutomationTier.RECOMMENDATION
        return AutomationTier.INFORMATIVE

    def _generate_recommendations(self, diagnosis: str, tier: AutomationTier) -> List[str]:
        recommendations = []
        diagnosis_table = self.DIAGNOSIS_ACTIONS.get(diagnosis)
        if diagnosis_table and tier in diagnosis_table:
            recommendations.extend(diagnosis_table[tier])
        else:
            recommendations.extend(self.GENERIC_ACTIONS[tier])
        return recommendations

    def evaluate(
        self,
        diagnosis: str,
        icd10_codes: List[str],
        ecg: Dict,
        echo: Optional[Dict] = None,
        lab: Optional[Dict] = None,
    ) -> AutomationDecision:
        """Full protocol: confidence -> tier -> tier-and-diagnosis-specific recommendations."""
        confidence = round(self._calculate_confidence(ecg, echo, lab), 2)
        tier = self._get_automation_tier(confidence, icd10_codes)
        recommendations = self._generate_recommendations(diagnosis, tier)
        return AutomationDecision(
            tier=tier,
            confidence=confidence,
            icd10_codes=icd10_codes,
            diagnosis=diagnosis,
            recommendations=recommendations,
            requires_human_signoff=(tier != AutomationTier.URGENT),
            auto_escalate=(tier == AutomationTier.URGENT),
            action_label=self.TIER_ACTION_LABEL[tier],
        )
