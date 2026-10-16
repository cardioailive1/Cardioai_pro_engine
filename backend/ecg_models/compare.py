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
            "warning": ("Few outcome events for a neural network; expect unstable results and prefer the "
                        "feature model unless this network clearly beats it.") if warn else None}


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
