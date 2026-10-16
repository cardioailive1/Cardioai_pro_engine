"""
Build ECG -> MACE training examples inside a Mayo Workspace.

Each input row is a dict: patient_id, waveform_b64, label_window (a
training.label_schema.LabelWindow value), optional age and sex_male. Only
early_30_90 (positive) and none_observed (negative) rows are kept; imminent
and late events are excluded, exactly like LabelValidator.early_detection_eval_set.
The waveform layout (dtype, n_leads, byte order, interleaved/lead-major) must
be passed explicitly - confirm it with Mayo first (ecg_waveform.probe_waveform).
"""
from __future__ import annotations

import numpy as np

from training.label_schema import LabelWindow
from training.mayo.ecg_waveform import decode_waveform


def fit_length(sig: np.ndarray, length: int) -> np.ndarray:
    """Centre-crop or zero-pad (n_leads, n) to (n_leads, length)."""
    n = sig.shape[1]
    if n >= length:
        s = (n - length) // 2
        return sig[:, s:s + length]
    out = np.zeros((sig.shape[0], length), dtype=sig.dtype)
    out[:, :n] = sig
    return out


def normalize(sig: np.ndarray) -> np.ndarray:
    """Per-lead z-score; removes gain/offset differences between devices."""
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True)
    return ((sig - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def build_examples(rows, *, length: int = 5000, **layout):
    """-> list of dicts: patient_id, x (n_leads, length), y (0/1), context (age/65, sex)."""
    out, skipped = [], {"excluded_window": 0, "bad_waveform": 0}
    for r in rows:
        w = LabelWindow(r["label_window"])
        if w not in (LabelWindow.EARLY_30_90, LabelWindow.NONE_OBSERVED):
            skipped["excluded_window"] += 1
            continue
        try:
            sig = decode_waveform(r["waveform_b64"], **layout)
        except ValueError:
            skipped["bad_waveform"] += 1
            continue
        out.append({
            "patient_id": r["patient_id"],
            "x": normalize(fit_length(sig, length)),
            "y": int(w == LabelWindow.EARLY_30_90),
            "context": np.array([r.get("age", 65) / 65.0, float(r.get("sex_male", 0))], dtype=np.float32),
        })
    return out, skipped
