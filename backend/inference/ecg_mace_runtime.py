"""
Live inference for the ECG -> MACE network. Used by inference/models.py's
load_trained_model("ecg") ONLY when an approved checkpoint exists in weights/
(see ecg_models/checkpoint.py). With none, load_trained_model returns None and
the placeholder keeps running, exactly as before.

Record: {"waveform": [[lead1 samples], [lead2 samples], ...], "sample_rate_hz": 500}.
A 12-lead record is routed to the 12-lead model, a 1-lead record to the single-lead
(wearable) model if one is approved; otherwise the caller falls back to the placeholder.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np
import torch

from ecg_models.checkpoint import load_checkpoint
from ecg_models.explain import explain

WEIGHTS_DIR = os.environ.get("CARDIOAI_WEIGHTS_DIR", "weights")
CHECKPOINTS = {12: "ecg_mace_v1.pt", 1: "ecg_mace_1lead_v1.pt"}


def _prep(wave: np.ndarray, rate: float, meta: dict) -> np.ndarray:
    if abs(rate - meta["sample_rate_hz"]) > 1e-6:                      # resample by linear interpolation
        n_new = int(round(wave.shape[1] * meta["sample_rate_hz"] / rate))
        wave = np.stack([np.interp(np.linspace(0, wave.shape[1] - 1, n_new), np.arange(wave.shape[1]), l) for l in wave])
    L = meta["length"]
    if wave.shape[1] >= L:
        s = (wave.shape[1] - L) // 2; wave = wave[:, s:s + L]
    else:
        pad = np.zeros((wave.shape[0], L), dtype=wave.dtype); pad[:, :wave.shape[1]] = wave; wave = pad
    mu, sd = wave.mean(1, keepdims=True), wave.std(1, keepdims=True)
    return ((wave - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def _tier(p: float, cut: dict) -> str:
    return "high" if p >= cut["high"] else "moderate" if p >= cut["moderate"] else "low"


def build_runtime(weights_dir: Optional[str] = None) -> Optional[Callable[[dict[str, Any]], Optional[dict[str, Any]]]]:
    weights_dir = weights_dir or WEIGHTS_DIR
    loaded = {n: load_checkpoint(os.path.join(weights_dir, f)) for n, f in CHECKPOINTS.items()}
    loaded = {n: v for n, v in loaded.items() if v}
    if not loaded:
        return None

    def run(record: dict[str, Any]) -> Optional[dict[str, Any]]:
        wf = record.get("waveform")
        if wf is None:
            return None                                    # no waveform -> caller uses placeholder
        wave = np.asarray(wf, dtype=np.float32)
        if wave.ndim != 2 or wave.shape[0] not in loaded:
            return None                                    # no approved model for this lead count
        model, meta = loaded[wave.shape[0]]
        x = _prep(wave, float(record.get("sample_rate_hz", meta["sample_rate_hz"])), meta)
        with torch.no_grad():
            logit = model(torch.tensor(x).unsqueeze(0)).item()
        p = float(1 / (1 + np.exp(-(meta["platt"]["a"] * logit + meta["platt"]["b"]))))
        return {"model": "ecg_mace_risk", "model_version": f"ecg-mace-{meta['n_leads']}lead-{meta['created']}",
                "score": round(p, 4), "risk_tier": _tier(p, meta["tier_cutoffs"]), "window_days": "30-90",
                "inputs_used": [f"raw_ecg_{meta['n_leads']}_lead"],
                "explanation": explain(model, x, meta["sample_rate_hz"]),
                "approved_by": meta["approved_by"], "trained_on": meta["trained_on"],
                "note": "Approved checkpoint; see its metadata file for validation results."}
    return run
