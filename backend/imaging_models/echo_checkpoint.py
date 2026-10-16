"""Checkpoints for the standalone echo LVEF model (MobileEchoNet). Same rule as the ECG network: the live
engine only loads a checkpoint a human approved, and approval refuses synthetic data or results that miss the
OKR KR 1.2 targets on a cardiologist-referenced validation set."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from imaging_models.mobilenet_echo_model import MobileEchoNet


def save_echo_checkpoint(model: MobileEchoNet, path: str, *, n_frames: int, img_size: int, trained_on: str, validation: dict):
    """validation: {n_studies, correlation, mae_lvef_points, kappa (optional), reference: 'cardiologist'}"""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {"variant": model.variant, "n_frames": n_frames, "img_size": img_size, "trained_on": trained_on,
            "validation": validation, "created": time.strftime("%Y-%m-%d"), "approved_for_live": False, "approved_by": None}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def approve_echo_checkpoint(path: str, approver: str, note: str, min_corr: float = 0.90, max_mae: float = 5.0, min_n: int = 100):
    """Defaults mirror OKR KR 1.2 (r >= 0.90, MAE <= 5 LVEF points); min_n is a floor we chose, adjust deliberately."""
    mp = Path(path).with_suffix(".json"); meta = json.loads(mp.read_text()); v = meta.get("validation") or {}
    if "synthetic" in meta["trained_on"].lower():
        raise ValueError("refusing to approve a checkpoint trained on synthetic data")
    if v.get("reference") != "cardiologist":
        raise ValueError("refusing to approve: validation must be against cardiologist-read LVEF")
    if v.get("n_studies", 0) < min_n:
        raise ValueError(f"refusing to approve: validation on {v.get('n_studies', 0)} studies, need at least {min_n}")
    if v.get("correlation", 0) < min_corr or v.get("mae_lvef_points", 1e9) > max_mae:
        raise ValueError(f"refusing to approve: needs correlation >= {min_corr} and MAE <= {max_mae}; got {v}")
    meta.update(approved_for_live=True, approved_by=approver, approval_note=note, approved_on=time.strftime("%Y-%m-%d"))
    mp.write_text(json.dumps(meta, indent=2))
    return meta


def load_echo_checkpoint(path: str):
    p = Path(path); mp = p.with_suffix(".json")
    if not (p.exists() and mp.exists()):
        return None
    meta = json.loads(mp.read_text())
    if not meta.get("approved_for_live"):
        return None
    model = MobileEchoNet(n_outputs=1, variant=meta["variant"])
    model.load_state_dict(torch.load(p, map_location="cpu")); model.eval()
    return model, meta
