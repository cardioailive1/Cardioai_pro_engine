"""Gates that stop the ECG network going live on weak evidence: calibration,
data-sufficiency warnings, and a paired comparison against the feature baseline."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def fit_platt(logits: np.ndarray, y: np.ndarray) -> dict:
    """Logistic recalibration p = sigmoid(a*logit + b), fit on VALIDATION data only."""
    if y.sum() == 0 or y.sum() == len(y):
        return {"a": 1.0, "b": 0.0, "note": "validation set has one class; not calibrated"}
    lr = LogisticRegression(C=1e6).fit(logits.reshape(-1, 1), y)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def data_sufficiency(n_pos_train: int, n_params: int) -> dict:
    """Rule-of-thumb warning, NOT a standard: a network with ~1e5 parameters
    trained on a few dozen events will overfit. Reports events and a flag."""
    warn = n_pos_train < 200
    return {"train_positive_events": n_pos_train, "model_parameters": n_params,
            "warning": ("Few outcome events for a flexible model; expect unstable results and prefer the simpler model unless this one clearly beats it.") if warn else None}


def paired_auc_delta(y, s_net, s_base, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUC(network) - AUC(baseline) on the SAME test patients with a bootstrap 95% CI.
    beats_baseline is True only if the whole interval is above zero."""
    y, s_net, s_base = map(np.asarray, (y, s_net, s_base))
    rng = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if y[i].sum() in (0, n):
            continue
        d.append(roc_auc_score(y[i], s_net[i]) - roc_auc_score(y[i], s_base[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = roc_auc_score(y, s_net) - roc_auc_score(y, s_base)
    return {"auc_network": float(roc_auc_score(y, s_net)), "auc_baseline": float(roc_auc_score(y, s_base)),
            "delta": float(point), "ci95": [float(lo), float(hi)], "beats_baseline": bool(lo > 0)}


def propose_tier_cutoffs(y_choose, p_choose, y_report, p_report, sens_targets=(0.90, 0.80, 0.60, 0.40)) -> list[dict]:
    """Candidate alert cutoffs for CLINICAL SIGN-OFF. Cutoffs are chosen on validation scores for each
    sensitivity target, then reported on the held-out test scores (sensitivity, specificity, PPV and the
    share of patients flagged). The clinical team picks the 'moderate' and 'high' cutoffs from this table."""
    y_choose, p_choose, y_report, p_report = map(np.asarray, (y_choose, p_choose, y_report, p_report))
    rows = []
    for t in sens_targets:
        pos = np.sort(p_choose[y_choose == 1])
        if len(pos) == 0:
            continue
        cut = float(np.quantile(pos, 1 - t))               # flags ~t of validation positives
        flag = p_report >= cut
        tp = int((flag & (y_report == 1)).sum()); fp = int((flag & (y_report == 0)).sum())
        fn = int((~flag & (y_report == 1)).sum()); tn = int((~flag & (y_report == 0)).sum())
        rows.append({"target_sensitivity": t, "cutoff": round(cut, 5),
                     "test_sensitivity": round(tp / max(tp + fn, 1), 3), "test_specificity": round(tn / max(tn + fp, 1), 3),
                     "test_ppv": round(tp / max(tp + fp, 1), 3), "share_flagged": round(float(flag.mean()), 3),
                     "test_positives": int(y_report.sum())})
    return rows
