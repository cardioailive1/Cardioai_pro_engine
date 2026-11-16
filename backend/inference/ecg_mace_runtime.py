"""
Live inference for the ECG -> MACE network. Used by inference/models.py's
load_trained_model("ecg") ONLY when an approved checkpoint exists in weights/
(see ecg_models/checkpoint.py). With none, load_trained_model returns None and
the placeholder keeps running, exactly as before.

Record: {"waveform": [[lead1 samples], ...] or {"I": [...], "II": [...], ...}, "sample_rate_hz": 250 or 500}.
8 leads (Mayo storage: I, II, V1-V6) are expanded to 12 first (inference/ecg_leads.py). A 12-lead record is routed to the 12-lead model, a 1-lead record to the single-lead
(wearable) model if one is approved; otherwise the caller falls back to the placeholder.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np
import torch

from ecg_models.checkpoint import load_checkpoint
from ecg_models.ecg_v2_stacker import calibrated_logit, load_v2, missing_inputs, predict_v2, recalibrated_probability
from ecg_models.explain import explain
from ecg_models.stage2_checkpoint import FILE_NAME as STAGE2_FILE, file_sha256, load_stage2, predict_stage2, valid_echo
from inference.ecg_leads import to_lead_array

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
    # Optional second stage (ECG score + earlier echo LVEF). Used only if approved AND fitted on THIS 12-lead checkpoint.
    stage2 = load_stage2(os.path.join(weights_dir, STAGE2_FILE))
    if stage2 and (12 not in loaded or stage2.get("ecg_checkpoint_sha256") != file_sha256(os.path.join(weights_dir, CHECKPOINTS[12]))):
        stage2 = None

    # Optional v2: the same 12-lead network plus the history stacker. Used only when fully approved and every checksum matches (ecg_models/ecg_v2_stacker.py).
    v2 = load_v2(weights_dir, os.path.join(weights_dir, CHECKPOINTS[12])) if 12 in loaded else None

    def run(record: dict[str, Any]) -> Optional[dict[str, Any]]:
        wf = record.get("waveform")
        if wf is None:
            return None                                    # no waveform -> caller uses placeholder
        try:
            wave = to_lead_array(wf)                       # 8 Mayo leads -> 12; dict or list accepted
        except ValueError:
            return None
        if wave.shape[0] not in loaded:
            return None                                    # no approved model for this lead count
        model, meta = loaded[wave.shape[0]]
        x = _prep(wave, float(record.get("sample_rate_hz", meta["sample_rate_hz"])), meta)
        with torch.no_grad():
            logit = model(torch.tensor(x).unsqueeze(0)).item()
        p = float(1 / (1 + np.exp(-(meta["platt"]["a"] * logit + meta["platt"]["b"]))))
        extra: dict = {}
        if v2 and wave.shape[0] == 12:
            miss = missing_inputs(v2, record)
            if not miss:
                p_raw = predict_v2(v2, record, calibrated_logit(logit, meta["platt"]))
                prob = recalibrated_probability(v2, p_raw)
                conds = list(v2["conditions"])
                return {"model": "ecg_mace_risk", "model_version": f"ecg-mace-v2-{v2['created']}", "score": round(p_raw, 4),
                        "risk_tier": _tier(p_raw, v2["tier_cutoffs"]), "window_days": "30-90",
                        "calibrated_probability": None if prob is None else round(prob, 4),
                        "probability_note": ("recalibrated to the natural event rate; not yet verified on newer data" if prob is not None
                                             else "no recalibration file loaded: the score ranks risk but is not a stated probability"),
                        "inputs_used": [f"raw_ecg_{meta['n_leads']}_lead", "age", "sex", "heart_rate", "qt", "qrs", "history_conditions", "visit_counts"],
                        "history_conditions_used": conds, "explanation": explain(model, x, meta["sample_rate_hz"]),
                        "approved_by": v2["approval"]["approved_by"], "trained_on": meta["trained_on"],
                        "second_stage": "not applied: the ECG + echo second stage was fitted on the v1 score",
                        "note": "ECG v2 (network + history stacker); approved per approval_v2.json; see the locked-set report."}
            extra = {"v2_not_used": "v2 needs these inputs and did not get them: " + ", ".join(miss) + "; scored with the v1 network"}
        if stage2 and wave.shape[0] == 12:
            echo = record.get("echo_lvef") or {}
            age, sex = record.get("age"), record.get("sex_male")
            if age is None or sex is None:
                extra = {"second_stage": "not used: the ECG + echo model needs age and sex on the record; scored by the ECG alone"}
            else:
                ok, why = valid_echo(echo.get("lvef"), echo.get("days_since_echo"), stage2["max_days_before"])
                ecg_only = p
                p = predict_stage2(stage2, ecg_only, ok, echo.get("lvef") if ok else None, float(age), bool(sex))
                meta = {**meta, "tier_cutoffs": stage2["tier_cutoffs"], "approved_by": stage2["approved_by"],
                        "trained_on": stage2["trained_on"], "created": stage2["created"]}
                extra = {"second_stage": "ECG + echo LVEF model", "ecg_only_score": round(ecg_only, 4), "echo_used": ok,
                         "echo_note": "" if ok else f"echo not used ({why}); scored as a patient with no earlier echo",
                         "inputs_used_extra": ["echo_lvef", "age", "sex"] if ok else ["age", "sex"]}
        inputs = [f"raw_ecg_{meta['n_leads']}_lead"] + extra.pop("inputs_used_extra", [])
        return {**extra, "model": "ecg_echo_mace_risk" if extra.get("echo_used") is not None else "ecg_mace_risk",
                "model_version": (f"ecg-echo-mace-{meta['created']}" if extra.get("echo_used") is not None else f"ecg-mace-{meta['n_leads']}lead-{meta['created']}"),
                "score": round(p, 4), "risk_tier": _tier(p, meta["tier_cutoffs"]), "window_days": "30-90",
                "inputs_used": inputs,
                "explanation": explain(model, x, meta["sample_rate_hz"]),
                "approved_by": meta["approved_by"], "trained_on": meta["trained_on"],
                "note": "Approved checkpoint; see its metadata file for validation results."}
    return run
