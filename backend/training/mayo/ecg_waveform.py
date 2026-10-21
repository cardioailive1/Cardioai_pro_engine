"""
CardioAI Pro - Mayo ECG waveform decoding
=========================================
Mayo Clinic Platform stores the ECG waveform table as a Base64-encoded
string that must be decoded before use (confirmed by Mayo, 2026-10-14).

CONFIRMED BY MAYO (Adam Resnick and Tommy Stretch, 2026-10-21) for FACT_ECG_WAVEFORM:
  * one WAVEFORM record per lead, URL-encoded Base64; decode each record on its own
    (`decode_mayo_lead`): URL-decode, Base64-decode, read as signed 16-bit little-endian
    counts, multiply by 4.88 for microvolts;
  * there is NO chunk order. The STRING_AGG on the slide does not define a sequence, so
    records are never concatenated (`assemble_mayo_ecg` refuses a lead with several records);
  * 8 LEAD_ID values (I, II, V1-V6); use WAVEFORM_TYPE = 'Rhythm' for the full 10-second ECG
    (the 'median' type is one synthesized beat);
  * sample rate is ECGSAMPLEBASE in FACT_ECG_WAVEFORM_RESTING_MEASUREMENT (250 or 500 Hz);
    `resample` brings every ECG to one rate.
Still unconfirmed: how the resting-measurement table joins to the waveform rows.

The older generic helpers below (`decode_waveform`, `probe_waveform`) take the layout as
explicit arguments and remain for other sources. A wrong layout produces plausible-looking
garbage, so always check a decoded trace by eye against the signed ECG report.
"""
from __future__ import annotations

import base64
import binascii
from typing import Iterable, Optional
from urllib.parse import unquote

import numpy as np

_DTYPES = {"int16": "i2", "int32": "i4", "float32": "f4", "float64": "f8"}

MAYO_LEAD_IDS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
MAYO_UV_PER_COUNT = 4.88
MAYO_FULL_RECORDING_TYPE = "rhythm"


def b64_to_bytes(s: str) -> bytes:
    s = "".join(s.split())
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s, validate=True)
    except binascii.Error:
        return base64.urlsafe_b64decode(s)


def decode_waveform(b64: str, *, dtype: str = "int16", n_leads: int = 12,
                    byteorder: str = "little", layout: str = "lead_major",
                    gain: Optional[float] = None) -> np.ndarray:
    """Return array shaped (n_leads, n_samples). `layout` is 'lead_major'
    (all samples of lead 1, then lead 2...) or 'interleaved' (sample 1 of
    every lead, then sample 2...). `gain` converts counts to mV if known."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if layout not in ("lead_major", "interleaved"):
        raise ValueError("layout must be 'lead_major' or 'interleaved'")
    code = ("<" if byteorder == "little" else ">") + _DTYPES[dtype]
    raw = b64_to_bytes(b64)
    arr = np.frombuffer(raw, dtype=np.dtype(code))
    if arr.size % n_leads:
        raise ValueError(f"{arr.size} samples is not divisible by {n_leads} leads; layout/dtype is wrong")
    out = arr.reshape(n_leads, -1) if layout == "lead_major" else arr.reshape(-1, n_leads).T
    out = out.astype(np.float32)
    return out * gain if gain else out


def probe_waveform(b64: str, lead_options=(1, 8, 12), sample_rates=(250, 500, 1000)) -> list[dict]:
    """List layouts whose decoded length/amplitude look physiologic (a
    10-second 12-lead is typical). Candidates only - a human confirms."""
    raw = b64_to_bytes(b64)
    found = []
    for dt, code in _DTYPES.items():
        size = np.dtype(code).itemsize
        if len(raw) % size:
            continue
        for bo in ("little", "big"):
            a = np.frombuffer(raw, dtype=np.dtype(("<" if bo == "little" else ">") + code))
            if not np.isfinite(a).all() or a.size == 0:
                continue
            for nl in lead_options:
                if a.size % nl:
                    continue
                secs = [a.size // nl / sr for sr in sample_rates]
                if any(2 <= s <= 30 for s in secs):
                    found.append({"dtype": dt, "byteorder": bo, "n_leads": nl,
                                  "seconds_at_rates": dict(zip(sample_rates, secs)),
                                  "abs_p99": float(np.percentile(np.abs(a), 99))})
    return found


# ---------------------------------------------------------------- Mayo FACT_ECG_WAVEFORM
def decode_mayo_lead(value: str) -> np.ndarray:
    """One WAVEFORM record (one lead) -> float32 microvolts. URL-decode, Base64-decode, int16 little-endian, x 4.88."""
    s = unquote(str(value or "").strip())
    if not s:
        raise ValueError("empty waveform value")
    raw = b64_to_bytes(s)
    if len(raw) == 0 or len(raw) % 2:
        raise ValueError(f"{len(raw)} bytes is not a whole number of 16-bit samples")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) * MAYO_UV_PER_COUNT


def assemble_mayo_ecg(records: Iterable[dict]) -> np.ndarray:
    """records: dicts with LEAD_ID, WAVEFORM and (optionally) WAVEFORM_TYPE, all for ONE ECG acquisition.
    Keeps only 'Rhythm' records when a type is given. Returns (8, n) microvolts in the order I, II, V1-V6.
    Refuses (ValueError) a missing lead, unequal lead lengths, or more than one record for a lead:
    Mayo defines no order between records, so they must not be joined."""
    by_lead: dict = {}
    for r in records:
        t = r.get("WAVEFORM_TYPE")
        if t is not None and str(t).strip().lower() != MAYO_FULL_RECORDING_TYPE:
            continue
        lead = str(r.get("LEAD_ID", "")).strip().upper()
        by_lead.setdefault(lead, []).append(r["WAVEFORM"])
    missing = [l for l in MAYO_LEAD_IDS if l not in by_lead]
    if missing:
        raise ValueError(f"missing leads {missing}")
    many = [l for l in MAYO_LEAD_IDS if len(by_lead[l]) > 1]
    if many:
        raise ValueError(f"more than one Rhythm record for leads {many}; Mayo defines no chunk order, inspect these rows")
    sigs = [decode_mayo_lead(by_lead[l][0]) for l in MAYO_LEAD_IDS]
    if len({s.size for s in sigs}) != 1:
        raise ValueError(f"lead lengths differ: {[s.size for s in sigs]}")
    return np.vstack(sigs)


def resample(sig: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """(leads, n) at src_hz -> (leads, n') at dst_hz. Whole-number downsampling averages blocks; everything else interpolates linearly."""
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError("sample rates must be positive")
    if src_hz == dst_hz:
        return sig
    if src_hz > dst_hz and (src_hz / dst_hz).is_integer():
        k, n = int(src_hz // dst_hz), sig.shape[1]
        return sig[:, : n - n % k].reshape(sig.shape[0], -1, k).mean(axis=2).astype(np.float32)
    n_new = int(round(sig.shape[1] * dst_hz / src_hz))
    x_old = np.arange(sig.shape[1]) / src_hz
    x_new = np.arange(n_new) / dst_hz
    return np.vstack([np.interp(x_new, x_old, row) for row in sig]).astype(np.float32)
