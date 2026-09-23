"""
CardioAI Pro — Multi-Modal Fusion
======================================
Combines the ECG risk score (inference/models.py) and the longitudinal
trend assessment (longitudinal/trend_engine.py) into one transparent,
auditable score. This is NOT a trained fusion model — no such model
exists — it's an explicit weighted combination with hand-set weights,
serving the same interface role load_trained_model() does elsewhere: the
seam a real learned fusion model plugs into once one exists, trained on
outcomes that actually span multiple modalities.

WHY IMAGING IS DELIBERATELY *NOT* FUSED INTO THE SCORE:
integrations/dicom.py's extract_pixel_features() produces real numbers
from real pixel data — intensity, entropy, gradient magnitude — but no
trained model exists to say what those numbers mean clinically (a high
gradient-magnitude mean could be a normal vessel border or a scan
artifact; nothing here can tell the difference). Folding an uninterpreted
number into a risk score would silently launder it into looking like
clinical signal, exactly the overclaiming this engine avoids everywhere
else. So imaging features are surfaced alongside the fused assessment,
clearly labeled as not-yet-interpreted, never blended into the number.

THE FUSION WEIGHTS ARE HAND-SET, NOT LEARNED:
0.75 ECG / 0.25 longitudinal-trend is a reasonable-sounding default, not a
validated one. A real fusion model would learn these weights (and almost
certainly a nonlinear combination) from outcomes data the way
training/train_model.py's candidate model learns the HR x HRV interaction
— see that module for the pattern this would follow once real multi-modal
outcome data exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ECG_WEIGHT = 0.75
TREND_WEIGHT = 0.25


@dataclass
class FusedAssessment:
    fused_score: float
    fused_tier: str
    ecg_score: Optional[float]
    ecg_contribution: float
    trend_risk: Optional[float]
    trend_contribution: float
    imaging_available: bool
    imaging_features: Optional[dict[str, Any]]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fused_score": round(self.fused_score, 4), "fused_tier": self.fused_tier,
            "ecg_score": self.ecg_score, "ecg_contribution": round(self.ecg_contribution, 4),
            "trend_risk": self.trend_risk, "trend_contribution": round(self.trend_contribution, 4),
            "imaging_available": self.imaging_available, "imaging_features": self.imaging_features,
            "narrative": self.narrative,
            "fusion_weights": {"ecg": ECG_WEIGHT, "longitudinal_trend": TREND_WEIGHT},
            "note": "Hand-set weights, not learned from outcomes — see module docstring. Imaging features (if present) are NOT included in fused_score.",
        }


def _tier_from_score(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "moderate"
    return "low"


class MultiModalFusionEngine:
    def fuse(
        self,
        ecg_prediction: Optional[dict[str, Any]],
        longitudinal_assessment: Optional[dict[str, Any]],
        imaging_features: Optional[dict[str, Any]] = None,
    ) -> FusedAssessment:
        ecg_score = ecg_prediction.get("score") if ecg_prediction else None
        trend_risk = (
            longitudinal_assessment.get("trend_risk_contribution")
            if longitudinal_assessment and longitudinal_assessment.get("sufficient_data")
            else None
        )

        if ecg_score is None and trend_risk is None:
            return FusedAssessment(
                fused_score=0.0, fused_tier="low", ecg_score=None, ecg_contribution=0.0,
                trend_risk=None, trend_contribution=0.0, imaging_available=imaging_features is not None,
                imaging_features=imaging_features, narrative="No ECG score or sufficient trend history available.",
            )

        # Renormalize weights when one input is missing, rather than silently
        # treating a missing signal as zero risk.
        if ecg_score is not None and trend_risk is not None:
            ecg_w, trend_w = ECG_WEIGHT, TREND_WEIGHT
        elif ecg_score is not None:
            ecg_w, trend_w = 1.0, 0.0
        else:
            ecg_w, trend_w = 0.0, 1.0

        ecg_contribution = ecg_w * (ecg_score or 0.0)
        trend_contribution = trend_w * (trend_risk or 0.0)
        fused_score = min(1.0, ecg_contribution + trend_contribution)
        fused_tier = _tier_from_score(fused_score)

        pieces = []
        if ecg_score is not None:
            pieces.append(f"ECG score {ecg_score:.3f} (weight {ecg_w:.2f})")
        if trend_risk is not None:
            pieces.append(f"trend risk {trend_risk:.3f} (weight {trend_w:.2f})")
        narrative = f"Fused from: {', '.join(pieces)}. " + (
            "Imaging features present but NOT fused (no trained interpretation model)." if imaging_features
            else "No imaging data for this record."
        )

        return FusedAssessment(
            fused_score=fused_score, fused_tier=fused_tier, ecg_score=ecg_score, ecg_contribution=ecg_contribution,
            trend_risk=trend_risk, trend_contribution=trend_contribution, imaging_available=imaging_features is not None,
            imaging_features=imaging_features, narrative=narrative,
        )
