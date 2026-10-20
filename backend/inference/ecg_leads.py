"""
Lead handling for the ECG -> MACE runtime and training.

Mayo's ECG slides (2026-10-19): only 8 leads are stored (I, II, V1-V6); the other four are calculated. We rebuild
the standard 12 with the usual limb-lead identities so the 12-lead model sees the same input in training and live:

    III = II - I      aVR = -(I + II) / 2      aVL = I - II / 2      aVF = II - I / 2

Order of the 12 outputs: I, II, III, aVR, aVL, aVF, V1..V6.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MAYO_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
TWELVE = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_CANON = {n.lower(): n for n in TWELVE}
_CANON.update({"lead_" + n.lower(): n for n in TWELVE})


def derive_twelve_lead(eight: np.ndarray) -> np.ndarray:
    """(8, n) in Mayo order I, II, V1..V6 -> (12, n) standard order."""
    e = np.asarray(eight, dtype=np.float32)
    if e.ndim != 2 or e.shape[0] != 8:
        raise ValueError(f"expected 8 leads (I, II, V1-V6), got shape {list(e.shape)}")
    i, ii, v = e[0], e[1], e[2:]
    return np.vstack([i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2, v]).astype(np.float32)


def to_lead_array(waveform: Any) -> np.ndarray:
    """Accepts a dict {lead_name: samples} or a list of rows. Returns (1, n) or (12, n).
    8 rows (or the 8 Mayo lead names) are expanded to 12; 12 rows pass through; 1 row is the single-lead case."""
    if isinstance(waveform, dict):
        named = {}
        for k, v in waveform.items():
            c = _CANON.get(str(k).strip().lower())
            if c is None:
                raise ValueError(f"unknown lead name '{k}'")
            named[c] = np.asarray(v, dtype=np.float32)
        if len({len(v) for v in named.values()}) != 1:
            raise ValueError("all leads must have the same number of samples")
        if set(MAYO_LEADS) <= set(named):
            return derive_twelve_lead(np.stack([named[n] for n in MAYO_LEADS]))
        if set(TWELVE) <= set(named):
            return np.stack([named[n] for n in TWELVE]).astype(np.float32)
        if set(named) == {"I"}:
            return named["I"][None, :]
        raise ValueError("send lead I alone, the 8 Mayo leads (I, II, V1-V6), or all 12")
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("waveform must be a list of lead rows")
    if arr.shape[0] == 8:
        return derive_twelve_lead(arr)
    if arr.shape[0] in (1, 12):
        return arr
    raise ValueError(f"{arr.shape[0]} leads: send 1, 8 (I, II, V1-V6) or 12")
