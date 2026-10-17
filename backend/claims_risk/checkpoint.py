"""Checkpoint with provenance for the claims-risk model. Same rules as the ECG network: the live engine only loads
an approved checkpoint; approval refuses synthetic data, no paired win over the baseline, a missing subgroup report,
or unset tier cutoffs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib


def save_claims_checkpoint(result: dict, path: str, *, trained_on: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], p)
    meta = {"platt": result["platt"], "trained_on": trained_on, "metrics": result["metrics"], "baseline_auc": result["baseline_auc"],
            "comparison_to_baseline": result["comparison_to_baseline"], "subgroup_auc": result["subgroup_auc"],
            "cutoff_proposals": result["cutoff_proposals"], "tier_cutoffs": None, "created": time.strftime("%Y-%m-%d"),
            "horizon_days": 365, "approved_for_live": False, "approved_by": None}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    return meta


def set_claims_tier_cutoffs(path: str, moderate: float, high: float, set_by: str, note: str):
    if not (0 <= moderate < high <= 1):
        raise ValueError("need 0 <= moderate < high <= 1")
    mp = Path(path).with_suffix(".json"); m = json.loads(mp.read_text())
    m["tier_cutoffs"] = {"moderate": moderate, "high": high, "set_by": set_by, "note": note}
    mp.write_text(json.dumps(m, indent=2)); return m


def approve_claims_checkpoint(path: str, approver: str, note: str):
    mp = Path(path).with_suffix(".json"); m = json.loads(mp.read_text())
    if "synthetic" in m["trained_on"].lower():
        raise ValueError("refusing to approve a checkpoint trained on synthetic data")
    if not (m.get("comparison_to_baseline") or {}).get("beats_baseline"):
        raise ValueError("refusing to approve: model does not beat the age/sex/comorbidity baseline")
    if not m.get("subgroup_auc"):
        raise ValueError("refusing to approve: no subgroup report")
    if not m.get("tier_cutoffs"):
        raise ValueError("refusing to approve: tier cutoffs not set")
    m.update(approved_for_live=True, approved_by=approver, approval_note=note, approved_on=time.strftime("%Y-%m-%d"))
    mp.write_text(json.dumps(m, indent=2)); return m


def load_claims_checkpoint(path: str):
    p = Path(path); mp = p.with_suffix(".json")
    if not (p.exists() and mp.exists()):
        return None
    m = json.loads(mp.read_text())
    return (joblib.load(p), m) if m.get("approved_for_live") else None
