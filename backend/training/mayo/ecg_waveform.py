"""
CardioAI Pro - Mayo ECG waveform decoding
=========================================
Mayo Clinic Platform stores the ECG waveform table as a Base64-encoded
string that must be decoded before use (confirmed by Mayo, 2026-10-14).

What Mayo has NOT told us yet: the binary layout inside the string (sample
dtype, byte order, number of leads, interleaved vs. lead-major). That is
deliberately NOT guessed here. `decode_waveform` takes the layout as
explicit arguments, and `probe_waveform` helps you find it from a few real
rows in the Workspace, then confirm with Mayo (ECG training in the Resource
Hub covers the format). Wrong layout produces plausible-looking garbage, so
always check a decoded trace by eye against the signed ECG report.
"""
from __future__ import annotations

import base64
import binascii
from typing import Optional

import numpy as np

_DTYPES = {"int16": "i2", "int32": "i4", "float32": "f4", "float64": "f8"}


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
