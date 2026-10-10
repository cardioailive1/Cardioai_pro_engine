"""
CardioAI Pro — Benchmark Model
===================================
A simple, interpretable logistic regression over the same three features
the engine already ingests (heart rate, QTc, HRV). This is the statistical
floor a candidate model has to clear — if a more complex model can't beat
this, the complexity isn't earning its keep.

IMPORTANT — what this is and isn't: this is a LOCAL statistical baseline,
fit on this system's own training data. It is NOT the established clinical
risk scores (HEART, TIMI, GRACE) used in real cardiology practice — those
require variables this engine doesn't currently ingest (troponin, clinical
history, ECG morphology beyond QTc, hemodynamics). A real deployment needs
head-to-head comparison against those too, on a cohort with the variables
to compute them; see the module docstring in train_model.py for what that
would require. Comparing against a fitted linear baseline is a necessary
first check, not a substitute for that external validation.

Reference points from cardiology literature, for calibrating expectations
(not something this local baseline is expected to match without troponin/
history data): the HEART score's reported AUC for 6-week MACE is
approximately 0.83 in its original derivation/validation studies; TIMI
risk score AUCs for 14-day MACE in ACS populations are more commonly in
the 0.65-0.75 range; GRACE score c-statistics for in-hospital or 6-month
outcomes are often reported around 0.82-0.86. These are commonly cited
figures from the validation literature for those specific instruments in
their specific populations — verify against the primary studies before
using any of them as a hard target, since populations and endpoints vary
study to study.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class BenchmarkModel:
    FEATURE_NAMES = ["heart_rate_bpm", "qt_interval_ms", "hrv_sdnn_ms"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def _to_matrix(self, records) -> np.ndarray:
        return np.array([[r.features[f] for f in self.FEATURE_NAMES] for r in records])

    def fit(self, records, labels: np.ndarray) -> "BenchmarkModel":
        X = self._to_matrix(records)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        return self

    def predict_proba(self, records) -> np.ndarray:
        X = self._to_matrix(records)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Interpretable weights — part of why a linear benchmark is worth having even when a fancier model wins on AUC."""
        return dict(zip(self.FEATURE_NAMES, self.model.coef_[0].round(4).tolist()))
