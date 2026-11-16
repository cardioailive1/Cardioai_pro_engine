"""
ECG v2: the frozen 12-lead network plus the history stacker (age, sex, heart rate, QT, QRS, 12 history conditions, visit counts).
Loaded by the live runtime ONLY when a folder weights/ecg_mace_v2/ holds all four files below AND every checksum matches AND a named
human approval is recorded. Any gap -> load_v2 returns None and the runtime keeps scoring with the v1 network exactly as before.

  frozen_ecg_v2.json   the freeze card written in the Mayo Workspace (feature list, qrs fill, checksums). Its own "approved" stays False
  stacker_v2.joblib    scaler + HistGradientBoosting model; a pickle, so it is opened ONLY after its SHA-256 matches the card and the approval
  approval_v2.json     who approved, when, the two tier cutoffs, and the stacker and network checksums that were approved
  recalibration_v2.json (optional) monotone map from the raw v2 score to a probability at natural prevalence; never changes a tier
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

FOLDER = "ecg_mace_v2"
HEAD = ["network_logit_calibrated", "age_years_div10", "sex_male", "heart_rate_bpm", "qt_interval_ms", "qrs_ms_filled"]
TAIL = ["log1p_acute_visits_365", "log1p_visits_365"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrated_logit(raw_logit: float, platt: dict) -> float:
    """Exactly the training-time transform: Platt, clipped to the same limits, back to a logit."""
    p = 1 / (1 + math.exp(-max(min(platt["a"] * raw_logit + platt["b"], 30.0), -30.0)))
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def load_v2(weights_dir: str, network_checkpoint_path: str) -> Optional[dict]:
    d = Path(weights_dir) / FOLDER
    names = ("frozen_ecg_v2.json", "stacker_v2.joblib", "approval_v2.json")
    if not all((d / n).exists() for n in names):
        return None
    try:
        card = json.loads((d / names[0]).read_text()); appr = json.loads((d / names[2]).read_text())
    except (OSError, ValueError):
        return None
    if not (appr.get("approved") is True and appr.get("approved_by")):
        return None                                                       # approval is a deliberate human step
    stacker_sha = _sha(d / names[1])
    if not (stacker_sha == card["stacker"]["sha256"] == appr.get("stacker_sha256")):
        return None
    net = Path(network_checkpoint_path)
    if not net.exists() or not (_sha(net) == card["network"]["sha256"] == appr.get("network_sha256")):
        return None                                                       # the stacker only makes sense with this exact network
    cut = appr.get("tier_cutoffs") or {}
    if not (0 <= float(cut.get("moderate", -1)) < float(cut.get("high", -1)) <= 1):
        return None
    try:
        import joblib
        bundle = joblib.load(d / names[1])
    except Exception:                                                     # sklearn/joblib missing or version problem -> v1 keeps running
        return None
    conds = list(card["history_features"]["conditions"])
    if list(bundle["features"]) != HEAD + conds + TAIL or list(card["stacker"]["features"]) != list(bundle["features"]):
        return None
    recal = None
    if (d / "recalibration_v2.json").exists():
        try:
            r = json.loads((d / "recalibration_v2.json").read_text())
            if r.get("stacker_sha256") == stacker_sha:
                recal = r
        except (OSError, ValueError):
            recal = None
    return {"bundle": bundle, "card": card, "approval": appr, "conditions": conds, "tier_cutoffs": {"moderate": float(cut["moderate"]), "high": float(cut["high"])},
            "recalibration": recal, "created": card["frozen_on"]}


def missing_inputs(v2: dict, record: dict[str, Any]) -> list[str]:
    miss = [k for k in ("age", "sex_male", "heart_rate_bpm", "qt_interval_ms", "n_acute_visits_365", "n_visits_365") if record.get(k) is None]
    hist = record.get("history") or {}
    miss += [f"history.{c}" for c in v2["conditions"] if hist.get(c) is None]      # unknown is not the same as absent: all 12 must be stated
    return miss


def predict_v2(v2: dict, record: dict[str, Any], z_cal: float) -> float:
    qrs = record.get("qrs_ms")
    qrs = float(v2["card"]["stacker"]["qrs_fill_median"]) if qrs is None else float(qrs)
    hist = record["history"]
    x = [z_cal, float(record["age"]) / 10, float(bool(record["sex_male"])), float(record["heart_rate_bpm"]), float(record["qt_interval_ms"]), qrs]
    x += [float(bool(hist[c])) for c in v2["conditions"]] + [math.log1p(float(record["n_acute_visits_365"])), math.log1p(float(record["n_visits_365"]))]
    b = v2["bundle"]
    return float(b["model"].predict_proba(b["scaler"].transform(np.array([x], dtype=float)))[:, 1][0])


def recalibrated_probability(v2: dict, p_raw: float) -> Optional[float]:
    r = v2.get("recalibration")
    if not r:
        return None
    p = min(max(p_raw, 1e-6), 1 - 1e-6); z = r["slope"] * math.log(p / (1 - p)) + r["intercept_at_natural_prevalence"]
    return float(1 / (1 + math.exp(-max(min(z, 30.0), -30.0))))
