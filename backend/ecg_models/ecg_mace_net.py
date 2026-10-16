"""
CardioAI Pro - ECG -> MACE network
==================================
Reads the raw ECG waveform and outputs one logit for "MACE event 30-90 days
after this ECG" - the same task, labels, cohort and evaluation as
training/train_model.py, so the two are directly comparable. It replaces
hand-made features (heart rate, QT, HRV) with features learned from the
trace, which avoids the feature-matching problems with Mayo's waveform table
(e.g. 10-second ECGs cannot give a meaningful SDNN).

A 1D residual CNN, not a transformer: for ECG-sized datasets this is the
proven choice and needs far less data. The architecture is NOT validated;
nothing here is trained on real data yet.

Input:  (B, n_leads, n_samples)  e.g. (B, 12, 5000) = 10 s at 500 Hz.
        Layout-agnostic: n_leads / length are arguments, so it runs once
        Mayo confirms the waveform format (see training/mayo/ecg_waveform.py).
Output: (B,) logits. Optional tabular context (age, sex) is concatenated to
        the pooled features when n_context > 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(self.drop(y)))
        return self.act(y + (x if self.skip is None else self.skip(x)))


class ECGMACENet(nn.Module):
    def __init__(self, n_leads: int = 12, n_context: int = 0, widths=(32, 64, 128, 128), dropout: float = 0.2):
        super().__init__()
        self.n_leads, self.n_context = n_leads, n_context
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(),
        )
        blocks, c = [], widths[0]
        for w in widths:
            blocks.append(_ResBlock(c, w, stride=2))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + n_context, 1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.blocks(self.stem(x)).mean(dim=-1)            # global average pool over time
        if self.n_context:
            if context is None:
                raise ValueError("model was built with n_context > 0; pass context")
            h = torch.cat([h, context], dim=1)
        return self.head(h).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
