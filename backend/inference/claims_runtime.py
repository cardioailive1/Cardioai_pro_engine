"""
Live scoring for payer members with the trained claims-risk model: 12-month probability of MI / stroke /
cardiovascular death. Loads weights/claims_risk_v1.joblib only if a human-approved checkpoint exists; otherwise
load_trained_model('claims') returns None and the placeholder keeps running.

The record must carry the member's diagnosis history (`history_diagnosis_codes`, ICD-10) and age; `sex` and
`inpatient_visits_12m` / `ed_visits_12m` are optional. A record without history codes gets the placeholder (a single
claim's own codes would understate the member's history and bias the score low).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from claims_risk.checkpoint import load_claims_checkpoint
from claims_risk.features import vectorize

WEIGHTS_DIR = os.environ.get("CARDIOAI_WEIGHTS_DIR", "weights")
CHECKPOINT = "claims_risk_v1.joblib"


def build_claims_runtime(weights_dir: Optional[str] = None):
    loaded = load_claims_checkpoint(os.path.join(weights_dir or WEIGHTS_DIR, CHECKPOINT))
    if not loaded:
        return None
    model, meta = loaded

    def run(record: dict[str, Any]) -> Optional[dict[str, Any]]:
        codes, age = record.get("history_diagnosis_codes"), record.get("member_age")
        if codes is None or not age:
            return None
        sex = record.get("sex")
        male = True if str(sex).upper() in ("M", "MALE") else False if str(sex).upper() in ("F", "FEMALE") else None
        x = np.array([vectorize(age, male, codes, record.get("inpatient_visits_12m"), record.get("ed_visits_12m"))])
        p = float(1 / (1 + np.exp(-np.clip(meta["platt"]["a"] * model.decision_function(x)[0] + meta["platt"]["b"], -30, 30))))
        c = meta["tier_cutoffs"]
        return {"model": "population_risk_stratifier", "model_version": f"claims-risk-{meta['created']}", "score": round(p, 4),
                "risk_tier": "high" if p >= c["high"] else "moderate" if p >= c["moderate"] else "low", "window_days": meta["horizon_days"],
                "inputs_used": ["age", "sex", "icd10_history", "visit_counts"], "approved_by": meta["approved_by"], "trained_on": meta["trained_on"],
                "note": "Approved checkpoint trained on Mayo OMOP history using claims-style inputs; validate on payer claims before relying on it."}
    return run
