"""
CardioAI Pro - second-stage MACE model: ECG network score + echo LVEF (+ age, sex)
==================================================================================
Heart failure (a low LVEF) changes MACE risk, and the ECG network may not fully capture it. This model adds the
patient's most recent EARLIER echo LVEF (training/mayo/ecg_echo_link.py) to the ECG network's score.

It is a small logistic regression, fitted on the VALIDATION patients' ECG scores (the ECG network never saw them) and
judged on the held-out TEST patients, so the extra input is judged fairly:
  * paired AUC difference against the ECG-only score on the SAME test patients, bootstrap 95% interval;
    `beats_ecg_only` is True only if the whole interval is above zero;
  * the same comparison on the patients who have an echo, because the gain can only come from them;
  * patients with no echo are scored by the same model (has_echo = 0, LVEF set to 0 after centering), so the model
    works when no echo exists.
This is NOT wired into the live engine. Live scoring would need a current LVEF on the patient's record, and the
engine reports echo LVEF as a standalone result today. Wiring is a separate step after this beats the ECG-only
model on real Mayo data and is approved.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ecg_models.compare import paired_auc_delta

FEATURES = ["ecg_logit", "has_echo", "lvef_centered", "lvef_le_40", "age_65", "sex_male"]
MIN_EVENTS_FOR_COMPARISON = 10


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stage2_matrix(examples, ecg_scores) -> np.ndarray:
    rows = []
    for e, s in zip(examples, ecg_scores):
        echo = e.get("echo") or {}
        has = bool(echo.get("has_echo"))
        lv = echo.get("lvef")
        ctx = e.get("context")
        rows.append([float(_logit(s)), float(has), ((lv - 55.0) / 10.0) if has else 0.0, float(has and lv <= 40.0),
                     float(ctx[0]) if ctx is not None else 1.0, float(ctx[1]) if ctx is not None else 0.0])
    return np.array(rows, dtype=float)


def fit_stage2(val_examples, val_ecg_scores, test_examples, test_ecg_scores, *, seed: int = 0) -> dict:
    """val/test_ecg_scores: the ECG network's calibrated probabilities for those examples (same order)."""
    yv = np.array([e["y"] for e in val_examples]); yt = np.array([e["y"] for e in test_examples])
    if yv.sum() < MIN_EVENTS_FOR_COMPARISON or yt.sum() < MIN_EVENTS_FOR_COMPARISON:
        return {"ok": False, "reason": f"need at least {MIN_EVENTS_FOR_COMPARISON} events in validation and test "
                                       f"(have {int(yv.sum())} and {int(yt.sum())})"}
    Xv, Xt = stage2_matrix(val_examples, val_ecg_scores), stage2_matrix(test_examples, test_ecg_scores)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xv, yv)
    p_test = lr.predict_proba(Xt)[:, 1]
    ecg_only = np.asarray(test_ecg_scores, dtype=float)
    has = np.array([bool((e.get("echo") or {}).get("has_echo")) for e in test_examples])
    overall = paired_auc_delta(yt, p_test, ecg_only, seed=seed)
    sub = None
    if has.sum() and yt[has].sum() >= MIN_EVENTS_FOR_COMPARISON and (~(yt[has] == 1)).sum() >= MIN_EVENTS_FOR_COMPARISON:
        sub = paired_auc_delta(yt[has], p_test[has], ecg_only[has], seed=seed)
    return {"ok": True, "features": FEATURES, "coefficients": dict(zip(FEATURES, map(float, lr.coef_[0]))),
            "intercept": float(lr.intercept_[0]), "n_test": int(len(yt)), "n_test_with_echo": int(has.sum()),
            "share_with_echo": float(has.mean()), "vs_ecg_only_all_test_patients": overall,
            "vs_ecg_only_patients_with_echo": sub, "beats_ecg_only": bool(overall["beats_baseline"]),
            "test_scores": p_test, "model": lr}
