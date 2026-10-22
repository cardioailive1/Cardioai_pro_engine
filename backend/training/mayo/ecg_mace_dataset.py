"""
Build ECG -> MACE training examples inside a Mayo Workspace.

Each input row is a dict: patient_id, label_window (a training.label_schema.LabelWindow
value), optional age and sex_male, and the waveform in ONE of two forms:
  * Mayo FACT_ECG_WAVEFORM (confirmed 2026-10-21): `leads` = {LEAD_ID: WAVEFORM value} for
    the 'Rhythm' recording, or `records` = the raw rows (LEAD_ID, WAVEFORM_TYPE, WAVEFORM),
    plus `sample_rate_hz` (ECGSAMPLEBASE). Every ECG is resampled to `target_rate_hz`.
  * legacy: `waveform_b64` with an explicit layout (other sources).
patient_id should be PATIENT_CLINIC_NUMBER (one person across record systems), so the
patient-level split cannot put one person on both sides. Only
early_30_90 (positive) and none_observed (negative) rows are kept; imminent
and late events are excluded, exactly like LabelValidator.early_detection_eval_set.
The waveform layout (dtype, n_leads, byte order, interleaved/lead-major) must
be passed explicitly - confirm it with Mayo first (ecg_waveform.probe_waveform).
"""
from __future__ import annotations

import numpy as np

from training.label_schema import LabelWindow
from inference.ecg_leads import derive_twelve_lead
from training.mayo.ecg_waveform import assemble_mayo_ecg, decode_waveform, resample


FEATURE_COLUMNS = ("heart_rate_bpm", "qt_interval_ms", "hrv_sdnn_ms")   # optional, for the baseline


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


def build_examples(rows, *, length: int = 5000, leads=None, target_rate_hz: float = 500.0, **layout):
    """-> list of dicts: patient_id, x (n_leads, length), y (0/1), context (age/65, sex).
    leads=[0] keeps one lead (e.g. lead I) to train the single-lead variant for wearables."""
    out, skipped = [], {"excluded_window": 0, "bad_waveform": 0}
    for r in rows:
        w = LabelWindow(r["label_window"])
        if w not in (LabelWindow.EARLY_30_90, LabelWindow.NONE_OBSERVED):
            skipped["excluded_window"] += 1
            continue
        try:
            if "leads" in r or "records" in r:
                recs = r.get("records") or [{"LEAD_ID": k, "WAVEFORM": v} for k, v in r["leads"].items()]
                sig = assemble_mayo_ecg(recs)
                if not r.get("sample_rate_hz"):
                    raise ValueError("sample_rate_hz (ECGSAMPLEBASE) is required")
                sig = resample(sig, float(r["sample_rate_hz"]), target_rate_hz)
            else:
                sig = decode_waveform(r["waveform_b64"], **layout)
            if sig.shape[0] == 8:                      # Mayo stores I, II, V1-V6 only; rebuild the standard 12
                sig = derive_twelve_lead(sig)
            if leads is not None:
                sig = sig[list(leads)]
        except ValueError:
            skipped["bad_waveform"] += 1
            continue
        out.append({
            "patient_id": r["patient_id"],
            "x": normalize(fit_length(sig, length)),
            "y": int(w == LabelWindow.EARLY_30_90),
            "echo": r.get("echo"),   # set by ecg_echo_link.link_ecg_to_echo; None when not linked
            "features": {k: float(r[k]) for k in FEATURE_COLUMNS if r.get(k) not in (None, "")},
            "context": np.array([r.get("age", 65) / 65.0, float(r.get("sex_male", 0))], dtype=np.float32),
        })
    return out, skipped
