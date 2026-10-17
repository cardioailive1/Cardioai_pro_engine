"""
Train the payer claims-risk model (12-month MACE from claims-style features).

    python -m training.train_claims_risk        (synthetic smoke test: proves plumbing, not performance)

On Mayo data: records = training.omop_claims_adapter.build_claims_risk_records(...), then
train_claims_risk(records). Patient-level split; gradient-boosted trees vs a logistic age / sex / comorbidity-count
baseline; Platt calibration on validation; paired AUC comparison; subgroup AUCs; cutoff proposals for the clinical /
actuarial team. The checkpoint is saved NOT approved (claims_risk/checkpoint.py).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from claims_risk.features import DX_GROUPS, FEATURE_NAMES, vectorize
from ecg_models.compare import data_sufficiency, fit_platt, paired_auc_delta, propose_tier_cutoffs

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_BASE_COLS = [_IDX["age"], _IDX["male"], _IDX["sex_known"]]
_GRP_COLS = [_IDX[g] for g in DX_GROUPS]


def split_by_patient(records, train_frac=0.7, val_frac=0.15, seed=23):
    rng = np.random.RandomState(seed)
    ids = sorted({r.patient_id for r in records}); rng.shuffle(ids)
    n = len(ids); tr, va = set(ids[:int(n * train_frac)]), set(ids[int(n * train_frac):int(n * (train_frac + val_frac))])
    te = set(ids) - tr - va
    pick = lambda s: [r for r in records if r.patient_id in s]
    return pick(tr), pick(va), pick(te)


def _X(recs):
    return np.array([r.x for r in recs]); 


def _baseline_X(X):
    return np.column_stack([X[:, _BASE_COLS], X[:, _GRP_COLS].sum(axis=1)])


def subgroup_auc(recs, scores, min_n=50, min_pos=5):
    out = {}
    for dim in ("gender", "age_band", "race_ethnicity"):
        for val in sorted({r.subgroups.get(dim) for r in recs if r.subgroups.get(dim)}):
            idx = [i for i, r in enumerate(recs) if r.subgroups.get(dim) == val]
            y = np.array([recs[i].label for i in idx])
            ok = len(idx) >= min_n and y.sum() >= min_pos and (len(y) - y.sum()) >= min_pos
            out[f"{dim}={val}"] = {"n": len(idx), "events": int(y.sum()), "auc": round(float(roc_auc_score(y, scores[idx])), 3) if ok else None}
    return out


def train_claims_risk(records, seed=23, n_estimators=150):
    train, val, test = split_by_patient(records, seed=seed)
    Xtr, Xva, Xte = _X(train), _X(val), _X(test)
    ytr, yva, yte = (np.array([r.label for r in s]) for s in (train, val, test))
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed).fit(Xtr, ytr)
    sc = StandardScaler().fit(_baseline_X(Xtr))
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(_baseline_X(Xtr)), ytr)
    platt = fit_platt(gb.decision_function(Xva), yva)
    cal = lambda X: 1 / (1 + np.exp(-np.clip(platt["a"] * gb.decision_function(X) + platt["b"], -30, 30)))
    s_te, s_va = cal(Xte), cal(Xva)
    b_te = lr.predict_proba(sc.transform(_baseline_X(Xte)))[:, 1]
    res = {"model": gb, "platt": platt, "n": {"train": len(train), "val": len(val), "test": len(test)},
           "metrics": {"auc": float(roc_auc_score(yte, s_te)), "average_precision": float(average_precision_score(yte, s_te)),
                       "brier": float(brier_score_loss(yte, s_te)), "base_rate": float(yte.mean())},
           "baseline_auc": float(roc_auc_score(yte, b_te)),
           "comparison_to_baseline": paired_auc_delta(yte, s_te, b_te),
           "subgroup_auc": subgroup_auc(test, s_te),
           "cutoff_proposals": propose_tier_cutoffs(yva, s_va, yte, s_te, sens_targets=(0.8, 0.6, 0.4, 0.2)),
           "sufficiency": data_sufficiency(int(ytr.sum()), n_estimators * 8)}
    return res


def synthetic_records(n=6000, seed=3):
    from training.omop_claims_adapter import ClaimsRiskRecord
    rng = np.random.RandomState(seed); out = []
    for i in range(n):
        age = rng.randint(40, 90); male = bool(rng.rand() < .5)
        codes = [c for c, p in (("I10", .45), ("E119", .2), ("N183", .08), ("I509", .06), ("I489", .07), ("I251", .1), ("E785", .3)) if rng.rand() < p]
        ip, ed = rng.poisson(.2), rng.poisson(.3)
        z = -3.6 + .045 * (age - 60) + .3 * male + .5 * ("I509" in codes) + .4 * ("I251" in codes) + .3 * ("E119" in codes) + .3 * ip
        out.append(ClaimsRiskRecord(f"p{i}", "2024-01-01", vectorize(age, male, codes, ip, ed), int(rng.rand() < 1 / (1 + np.exp(-z))),
                                    {"gender": "Male" if male else "Female", "age_band": "<50" if age < 50 else "50-64" if age < 65 else "65+"}))
    return out


if __name__ == "__main__":
    r = train_claims_risk(synthetic_records())
    print("SYNTHETIC data:", r["n"], r["metrics"], "baseline AUC", round(r["baseline_auc"], 3))
    print(r["comparison_to_baseline"]); print(r["sufficiency"])
