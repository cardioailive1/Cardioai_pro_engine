"""Occlusion explanations for the ECG network: how much does the risk score
drop when a time window (all leads) or a whole lead is blanked out? It shows
where the model looked. It is not a causal or clinical explanation."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def explain(model, x: np.ndarray, sample_rate_hz: float, window_s: float = 0.5, top_k: int = 3) -> dict:
    """x: (n_leads, n_samples), already normalized. Returns top windows and per-lead importance
    as the DROP in the model's logit when that part of the signal is zeroed."""
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    base = model(xt).item()
    w = max(1, int(window_s * sample_rate_hz))
    wins = []
    for s in range(0, x.shape[1], w):
        o = xt.clone(); o[:, :, s:s + w] = 0
        wins.append((s / sample_rate_hz, min(s + w, x.shape[1]) / sample_rate_hz, base - model(o).item()))
    leads = []
    for l in range(x.shape[0]):
        o = xt.clone(); o[:, l, :] = 0
        leads.append(base - model(o).item())
    top = sorted(wins, key=lambda t: -abs(t[2]))[:top_k]
    return {"method": "occlusion", "window_s": window_s,
            "top_windows": [{"start_s": round(a, 2), "end_s": round(b, 2), "logit_drop": round(d, 3)} for a, b, d in top],
            "lead_importance": [round(v, 3) for v in leads],
            "caveat": "Shows where the model looked, not why the patient is at risk."}
