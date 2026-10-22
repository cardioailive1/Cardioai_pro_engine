"""
What is actually scoring right now? One honest answer per model, read from the checkpoint files (the JSON sidecars)
without importing torch, so it works in the live service even when the model runtime is not installed.

state values:
  placeholder        no checkpoint in weights/ -> the hand-set heuristic runs (not a trained model)
  trained_not_approved   a checkpoint exists but no human has approved it -> still placeholder
  runtime_missing    approved checkpoint, but torch / scikit-learn is not installed in this service -> still placeholder
  live               approved and loadable -> the trained model scores
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

WEIGHTS_DIR = os.environ.get("CARDIOAI_WEIGHTS_DIR", "weights")

# key -> (label, file, runtime packages it needs, what it feeds)
MODELS = {
    "ecg_12lead": ("ECG to MACE, 12-lead", "ecg_mace_v1.pt", ("torch",), "clinician"),
    "ecg_1lead": ("ECG to MACE, single-lead (wearable proxy)", "ecg_mace_1lead_v1.pt", ("torch",), "clinician"),
    "echo_lvef": ("Echo LVEF (standalone, not in the MACE score)", "echo_lvef_v1.pt", ("torch",), "clinician"),
    "ecg_echo_stage2": ("ECG + echo LVEF second stage (adds the earlier echo to the ECG score)", "ecg_echo_stage2_v1.json", (), "clinician"),
    "claims_risk": ("Payer claims risk", "claims_risk_v1.joblib", ("sklearn", "joblib"), "payer"),
}


def _have(pkgs) -> list[str]:
    return [p for p in pkgs if importlib.util.find_spec(p) is None]


def _one(key: str, weights_dir: str) -> dict[str, Any]:
    label, fname, pkgs, surface = MODELS[key]
    ck = Path(weights_dir) / fname
    side = ck.with_suffix(".json")
    out: dict[str, Any] = {"key": key, "label": label, "surface": surface, "checkpoint": fname}
    if not (ck.exists() and side.exists()):
        who = "Nothing is scoring; echo shows 'awaiting trained model'." if key == "echo_lvef" else "The hand-set placeholder is scoring; it is not a trained model."
        return {**out, "state": "placeholder", "detail": f"No trained checkpoint in the weights folder. {who}"}
    try:
        meta = json.loads(side.read_text())
    except (OSError, ValueError):
        return {**out, "state": "placeholder", "detail": "Checkpoint metadata could not be read; placeholder is scoring."}
    out.update(trained_on=meta.get("trained_on"), created=meta.get("created"), approved_by=meta.get("approved_by"),
               cutoffs_set=bool(meta.get("tier_cutoffs")), metrics=meta.get("metrics") or meta.get("validation"))
    if not meta.get("approved_for_live"):
        return {**out, "state": "trained_not_approved", "detail": "A checkpoint exists but no one has approved it. The placeholder is still scoring."}
    missing = _have(pkgs)
    if missing:
        return {**out, "state": "runtime_missing", "detail": f"Approved, but this service does not have {', '.join(missing)} installed, so the placeholder is still scoring. Build with the model runtime (see RUNBOOK)."}
    return {**out, "state": "live", "detail": f"Trained model is scoring. Approved by {meta.get('approved_by')}."}


def model_status(weights_dir: str | None = None) -> dict[str, Any]:
    wd = weights_dir or WEIGHTS_DIR
    models = [_one(k, wd) for k in MODELS]
    return {"weights_dir": wd, "models": models,
            "any_live": any(m["state"] == "live" for m in models),
            "note": "Anything not 'live' is the hand-set placeholder, which is not clinically validated."}


def is_placeholder(prediction: dict | None) -> bool:
    from inference.models import MODEL_VERSION
    return bool(prediction) and prediction.get("model_version") == MODEL_VERSION
