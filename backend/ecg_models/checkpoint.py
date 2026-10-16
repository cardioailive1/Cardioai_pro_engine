"""Checkpoints with provenance. A checkpoint is only used by the live engine when a
human has approved it AND it beat the feature baseline; synthetic-trained files never qualify."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ecg_models.ecg_mace_net import ECGMACENet


def save_checkpoint(model: ECGMACENet, path: str, *, sample_rate_hz: float, length: int, platt: dict,
                    trained_on: str, metrics: dict, comparison: dict | None = None, tier_cutoffs: dict | None = None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {"n_leads": model.n_leads, "n_context": model.n_context, "sample_rate_hz": sample_rate_hz,
            "length": length, "platt": platt, "trained_on": trained_on, "metrics": metrics,
            "comparison_to_baseline": comparison, "tier_cutoffs": tier_cutoffs,
            "created": time.strftime("%Y-%m-%d"), "approved_for_live": False, "approved_by": None}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def approve_checkpoint(path: str, approver: str, note: str):
    """Deliberate human step. Refuses synthetic data or a model that did not beat the baseline."""
    mp = Path(path).with_suffix(".json"); meta = json.loads(mp.read_text())
    cmp_ = meta.get("comparison_to_baseline") or {}
    if "synthetic" in meta["trained_on"].lower():
        raise ValueError("refusing to approve a checkpoint trained on synthetic data")
    if not cmp_.get("beats_baseline"):
        raise ValueError("refusing to approve: no paired comparison showing the network beats the feature baseline")
    if not meta.get("tier_cutoffs"):
        raise ValueError("refusing to approve: risk-tier cutoffs not set (clinical team must set them)")
    meta.update(approved_for_live=True, approved_by=approver, approval_note=note, approved_on=time.strftime("%Y-%m-%d"))
    mp.write_text(json.dumps(meta, indent=2))
    return meta


def load_checkpoint(path: str):
    """-> (model, meta) or None if missing / not approved."""
    p = Path(path); mp = p.with_suffix(".json")
    if not (p.exists() and mp.exists()):
        return None
    meta = json.loads(mp.read_text())
    if not meta.get("approved_for_live"):
        return None
    model = ECGMACENet(n_leads=meta["n_leads"], n_context=meta["n_context"])
    model.load_state_dict(torch.load(p, map_location="cpu")); model.eval()
    return model, meta
