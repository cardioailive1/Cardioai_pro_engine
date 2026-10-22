"""
The ECG + echo (LVEF) second-stage MACE model: checkpoint, approval and scoring. Pure Python (no torch), so the live
service and the status page can read it without the model runtime.

The second stage takes the ECG network's calibrated probability and adds the most recent EARLIER echo LVEF. Its
coefficients only make sense with the exact ECG network that produced the scores it was fitted on, so the file records
that network's SHA-256 and is ignored if the live ECG checkpoint is a different file. Like every checkpoint it is used
live only after a human approves it, and approval is refused for synthetic data, no paired win over the ECG-only
score, or unset risk-tier cutoffs.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Optional

FILE_NAME = "ecg_echo_stage2_v1.json"
FEATURES = ["ecg_logit", "has_echo", "lvef_centered", "lvef_le_40", "age_65", "sex_male"]
LVEF_MIN, LVEF_MAX = 10.0, 80.0


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def features(ecg_prob: float, has_echo: bool, lvef: Optional[float], age_65: float, sex_male: float) -> list[float]:
    """The one place the second-stage inputs are built, shared by training and live scoring."""
    has = bool(has_echo) and lvef is not None
    return [logit(ecg_prob), float(has), ((lvef - 55.0) / 10.0) if has else 0.0, float(has and lvef <= 40.0), float(age_65), float(sex_male)]


def save_stage2(path: str, *, coefficients: dict, intercept: float, trained_on: str, comparison: dict,
                ecg_checkpoint_path: str, max_days_before: int = 365, metrics: Optional[dict] = None) -> dict:
    meta = {"features": FEATURES, "coefficients": {k: float(coefficients[k]) for k in FEATURES}, "intercept": float(intercept),
            "trained_on": trained_on, "comparison_to_ecg_only": comparison, "metrics": metrics,
            "ecg_checkpoint_sha256": file_sha256(ecg_checkpoint_path), "max_days_before": int(max_days_before),
            "tier_cutoffs": None, "created": time.strftime("%Y-%m-%d"), "approved_for_live": False, "approved_by": None}
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))
    return meta


def set_stage2_cutoffs(path: str, moderate: float, high: float, set_by: str, note: str) -> dict:
    if not (0 <= moderate < high <= 1):
        raise ValueError("need 0 <= moderate < high <= 1")
    p = Path(path); meta = json.loads(p.read_text())
    meta["tier_cutoffs"] = {"moderate": moderate, "high": high, "set_by": set_by, "note": note}
    p.write_text(json.dumps(meta, indent=2))
    return meta


def approve_stage2(path: str, approver: str, note: str) -> dict:
    p = Path(path); meta = json.loads(p.read_text())
    if "synthetic" in meta["trained_on"].lower():
        raise ValueError("refusing to approve a second-stage model trained on synthetic data")
    if not (meta.get("comparison_to_ecg_only") or {}).get("beats_baseline"):
        raise ValueError("refusing to approve: no paired comparison showing the ECG + echo model beats the ECG-only score")
    if not meta.get("tier_cutoffs"):
        raise ValueError("refusing to approve: risk-tier cutoffs not set (clinical team must set them)")
    meta.update(approved_for_live=True, approved_by=approver, approval_note=note, approved_on=time.strftime("%Y-%m-%d"))
    p.write_text(json.dumps(meta, indent=2))
    return meta


def load_stage2(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return meta if meta.get("approved_for_live") else None


def predict_stage2(meta: dict, ecg_prob: float, has_echo: bool, lvef: Optional[float], age: float, sex_male: bool) -> float:
    x = features(ecg_prob, has_echo, lvef, age / 65.0, 1.0 if sex_male else 0.0)
    z = meta["intercept"] + sum(meta["coefficients"][n] * v for n, v in zip(FEATURES, x))
    return float(1 / (1 + math.exp(-max(min(z, 30.0), -30.0))))


def valid_echo(lvef, days_since_echo, max_days_before: int) -> tuple[bool, str]:
    """Is this echo usable? Only an echo ON OR BEFORE the ECG, not older than the training window, with a plausible LVEF."""
    if lvef is None or days_since_echo is None:
        return False, "no echo supplied"
    if not (LVEF_MIN <= float(lvef) <= LVEF_MAX):
        return False, f"LVEF {lvef} is outside {LVEF_MIN:.0f}-{LVEF_MAX:.0f}%"
    if not (0 <= int(days_since_echo) <= max_days_before):
        return False, f"echo is {days_since_echo} days before the ECG; only 0-{max_days_before} days is used"
    return True, ""
