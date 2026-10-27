"""
CardioAI Pro - reviewed LVEF labels -> training data
====================================================
Connects the Mayo pipeline end to end:
  radiology_link.link_reports_to_images  (report <-> DICOM, accession number)
  report_labels / review_queue           (automatic labels, human-reviewed)
  lvef_dataset (this file)               (-> training/train_mobile_echo.py)

ONLY human-reviewed labels are used: `apply_reviews` output. Unreviewed or
excluded studies never reach training.

Honest limits, to settle once real data is in the Workspace:
  * Which series of a study is the cine to use (e.g. short-axis cine) is a
    caller decision - pass `series_uid_for` or select via SERIES_DESCRIPTION.
  * Frames are sorted by InstanceNumber and sampled evenly. A multi-slice
    short-axis stack needs a real temporal/slice ordering; this treats the
    chosen series as a single cine loop.
  * Target is LVEF/100 (0-1). The model is trained as a regression; this is
    a plumbing step, not a validated model.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from training.mayo.audit_gate import require_audited_labels


def build_lvef_examples(linked: dict, reviewed: dict, series_uid_for: Optional[Callable] = None) -> list[dict]:
    """linked: output['linked'] of link_reports_to_images. reviewed: apply_reviews(...)['labels'].
    One example per accession with a reviewed LVEF and at least one image instance."""
    require_audited_labels(reviewed)     # only labels that went through the automatic audit and both review tiers
    out = []
    for acc, item in linked.items():
        lab = reviewed.get(acc)
        if not lab:
            continue
        for study_uid, insts in item["studies"].items():
            series = {}
            for i in insts:
                series.setdefault(i["series"], []).append(i["sop"])
            if series_uid_for:
                uid = series_uid_for(study_uid, list(series))
                if uid is None:
                    continue
                chosen = series.get(uid)
            else:
                uid = max(series, key=lambda k: len(series[k]))  # default: the series with most instances
                chosen = series[uid]
            if chosen:
                out.append({"accession": acc, "study_uid": study_uid, "series_uid": uid,
                            "sops": [s for s in chosen if s], "lvef": float(lab["lvef"])})
    return out


class DicomCineLVEFDataset(Dataset):
    """path_for_sop: callable(sop_instance_uid) -> DICOM file path (where the Workspace
    puts the files you requested). Yields (video (T,1,H,W) in 0-1, target (1,) = LVEF/100)."""

    def __init__(self, examples: list[dict], path_for_sop: Callable[[str], str], n_frames: int = 8, img_size: int = 224):
        self.ex, self.path_for_sop, self.n_frames, self.img_size = examples, path_for_sop, n_frames, img_size

    def __len__(self) -> int:
        return len(self.ex)

    def _load(self, paths: list[str]) -> np.ndarray:
        import pydicom
        frames = []
        for p in paths:
            ds = pydicom.dcmread(p)
            a = ds.pixel_array.astype(np.float32)
            if a.ndim == 3:            # multi-frame instance
                frames.extend(list(a))
            else:
                frames.append((int(getattr(ds, "InstanceNumber", 0) or 0), a))
        # single-frame instances carry InstanceNumber for ordering
        tagged = [f for f in frames if isinstance(f, tuple)]
        if tagged:
            tagged.sort(key=lambda t: t[0])
            frames = [t[1] for t in tagged]
        return np.stack(frames)

    def __getitem__(self, idx: int):
        e = self.ex[idx]
        vol = self._load([self.path_for_sop(s) for s in e["sops"]])
        t = torch.from_numpy(vol)
        lo, hi = float(t.min()), float(t.max())
        t = (t - lo) / (hi - lo + 1e-6)
        sel = torch.linspace(0, len(t) - 1, self.n_frames).round().long()
        t = t[sel].unsqueeze(1)                                   # (T,1,H,W)
        t = torch.nn.functional.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return t, torch.tensor([e["lvef"] / 100.0], dtype=torch.float32)
