"""
CardioAI Pro — Model Registry & Inference Interface
=======================================================
IMPORTANT: No models are trained here. This module defines the *inference
interface* each modality's model will be called through — request shape,
response shape, versioning, routing — so that dropping in a real trained
model later is a one-function change (see `load_trained_model` below),
not a redesign of the API or orchestrator.

Until real weights exist, `_placeholder_*` functions compute a transparent,
clinically-motivated heuristic score purely so the rest of the system
(alerts, FHIR mapping, dashboard) has something real to flow through end
to end. These heuristics are NOT validated, NOT diagnostic, and must never
be presented to a clinician or consumer as a real risk prediction.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

MODEL_VERSION = "0.0.0-untrained-placeholder"


def load_trained_model(modality: str) -> Optional[Callable]:
    """
    Hook for real models. Once training artifacts exist, load them here,
    e.g.:

        if modality == "ecg":
            import torch
            model = torch.jit.load("weights/ecg_mace_v1.pt")
            model.eval()
            return lambda record: model(_featurize_ecg(record))

        if modality == "imaging":
            import torch
            from imaging_models.ct_model import CardiacCTViT
            model = CardiacCTViT(n_classes=4, patch_size=8, depth=12)
            model.load_state_dict(torch.load("weights/cardiac_ct_vit_v1.pt"))
            model.eval()
            return lambda dicom_meta: _run_ct_model(model, dicom_meta)
            # or, for echo specifically, choose by deployment context:
            #   imaging_models.echo_model.EchoVideoViT      — cloud/hospital,
            #     max accuracy (94.9M params, ~1.8s/inference on CPU)
            #   imaging_models.mobilenet_echo_model.MobileEchoNet — edge/
            #     point-of-care handheld ultrasound (8.5M params, ~100ms —
            #     17.8x faster; matches real published precedent for
            #     Raspberry Pi/smartphone echo deployment, see that
            #     module's docstring). CT stays on the ViT path — no
            #     equivalent edge-deployment driver exists for CT.

    Returning None (the default) means the registry falls back to the
    placeholder heuristic below.

    IMPORTANT — about the "window_days" field below: for the placeholder,
    it's a hardcoded label, not a validated claim. A real model wired in
    here must have been trained and evaluated against training/label_schema.py's
    MACELabelRecord spec — specifically its early_detection_eval_set(), which
    excludes imminent (0-30 day) events so the model can't pass by detecting
    events that are already unfolding — before "window_days" here means
    anything. See that module's docstring for why this distinction matters.
    """
    return None


def _risk_tier(score: float) -> str:
    if score >= 0.60:
        return "high"
    if score >= 0.30:
        return "moderate"
    return "low"


def _placeholder_ecg_risk(record: dict[str, Any]) -> dict[str, Any]:
    hr = record.get("heart_rate_bpm", 70) or 70
    qt = record.get("qt_interval_ms", 400) or 400
    hrv = record.get("hrv_sdnn_ms", 50) or 50

    # Transparent, bounded heuristic — NOT a trained model.
    score = 0.0
    score += max(0.0, (hr - 100)) * 0.01       # tachycardia contribution
    score += max(0.0, (qt - 460)) * 0.004        # prolonged QTc contribution
    score += max(0.0, (30 - hrv)) * 0.01         # depressed HRV contribution
    score = round(min(score, 1.0), 3)

    return {
        "model": "ecg_mace_risk", "model_version": MODEL_VERSION,
        "score": score, "risk_tier": _risk_tier(score), "window_days": 60,
        "inputs_used": ["heart_rate_bpm", "qt_interval_ms", "hrv_sdnn_ms"],
        "note": "Placeholder heuristic pending trained model — not clinically validated.",
    }


def _placeholder_wearable_score(record: dict[str, Any]) -> dict[str, Any]:
    hr = record.get("heart_rate_bpm", 70) or 70
    spo2 = record.get("spo2_pct", 98) or 98

    score = 0.0
    score += max(0.0, (hr - 110)) * 0.008
    score += max(0.0, (94 - spo2)) * 0.03
    score = round(min(score, 1.0), 3)

    return {
        "model": "personal_cardiac_score", "model_version": MODEL_VERSION,
        "score": score, "risk_tier": _risk_tier(score), "window_days": 90,
        "inputs_used": ["heart_rate_bpm", "spo2_pct"],
        "note": "Placeholder heuristic pending trained model — not clinically validated.",
    }


def _placeholder_population_risk(record: dict[str, Any]) -> dict[str, Any]:
    age = record.get("member_age", 50) or 50
    flags = record.get("risk_flags_count", 0) or 0

    score = round(min(0.01 * max(0, age - 40) + 0.08 * flags, 1.0), 3)

    return {
        "model": "population_risk_stratifier", "model_version": MODEL_VERSION,
        "score": score, "risk_tier": _risk_tier(score),
        "inputs_used": ["member_age", "risk_flags_count"],
        "note": "Placeholder heuristic pending trained model — not clinically validated.",
    }


class ModelRegistry:
    """Routes a (modality, record) pair to the right inference function."""

    def __init__(self):
        self._trained = {m: load_trained_model(m) for m in ("ecg", "wearable", "claims", "imaging")}
        self._placeholders: dict[str, Callable[[dict], dict]] = {
            "ecg": _placeholder_ecg_risk,
            "wearable": _placeholder_wearable_score,
            "claims": _placeholder_population_risk,
        }

    def predict(self, modality: str, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        if modality == "imaging":
            return None  # imaging inference point below

        trained = self._trained.get(modality)
        if trained is not None:
            return trained(record)

        fn = self._placeholders.get(modality)
        return fn(record) if fn else None

    def predict_imaging(self, dicom_meta: dict[str, Any]) -> dict[str, Any]:
        """
        Inference point for the imaging modality (echo/CT). The MODEL
        ARCHITECTURE now exists — imaging_models/ct_model.py's CardiacCTViT
        and imaging_models/echo_model.py's EchoVideoViT, both real, tested
        PyTorch code following published architectures (DINO-LG/CARD-ViT
        for CT, Echo-Vision-FM for echo) — but no pretrained WEIGHTS do.
        Self-supervised pretraining on a real unlabeled imaging corpus,
        which this environment has neither the data nor the compute
        budget to run, is what's still missing. Returns a stub result
        marking the endpoint as awaiting those weights.
        """
        return {
            "model": "imaging_mace_risk", "model_version": MODEL_VERSION,
            "status": "awaiting_trained_model",
            "note": "Architecture ready (see imaging_models/) — awaiting pretrained weights. Wire via load_trained_model('imaging').",
        }
