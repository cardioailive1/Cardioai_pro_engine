"""
Standalone echo LVEF inference. NOT part of the fused MACE score: the result is labeled so and is never
passed to multimodal_fusion. Loads weights/echo_lvef_v1.pt only if a human-approved checkpoint exists.
Preprocessing matches training (training/mayo/lvef_dataset.py): per-study min-max scaling, n_frames sampled
evenly across the loop, resized to img_size x img_size.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from imaging_models.echo_checkpoint import load_echo_checkpoint
from training.mayo.report_labels import ef_category

WEIGHTS_DIR = os.environ.get("CARDIOAI_WEIGHTS_DIR", "weights")
CHECKPOINT = "echo_lvef_v1.pt"
MIN_FRAMES = 8


def preprocess(frames: np.ndarray, n_frames: int, img_size: int) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    lo, hi = float(t.min()), float(t.max())
    t = (t - lo) / (hi - lo + 1e-6)
    sel = torch.linspace(0, len(t) - 1, n_frames).round().long()
    t = t[sel].unsqueeze(1)
    t = torch.nn.functional.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t.unsqueeze(0)                                      # (1, T, 1, H, W)


def build_echo_runtime(weights_dir: Optional[str] = None):
    loaded = load_echo_checkpoint(os.path.join(weights_dir or WEIGHTS_DIR, CHECKPOINT))
    if not loaded:
        return None
    model, meta = loaded

    def run(frames: np.ndarray) -> dict[str, Any]:
        frames = np.asarray(frames)
        if frames.ndim != 3 or frames.shape[0] < MIN_FRAMES:
            return {"status": "not_run", "reason": f"need a cine loop of at least {MIN_FRAMES} frames, got shape {list(frames.shape)}",
                    "fused_into_risk_score": False}
        with torch.no_grad():
            lvef = float(model(preprocess(frames, meta["n_frames"], meta["img_size"])).item()) * 100.0
        lvef = round(min(max(lvef, 5.0), 85.0), 1)
        return {"status": "ok", "model": "echo_lvef", "model_version": f"mobileechonet-{meta['variant']}-{meta['created']}",
                "lvef_percent": lvef, "lvef_category": ef_category(lvef),
                "fused_into_risk_score": False, "approved_by": meta["approved_by"], "trained_on": meta["trained_on"],
                "validation": meta["validation"],
                "note": "Standalone echo result. Not part of the MACE score. Validated on cart-based echo only unless the validation says otherwise."}
    return run
