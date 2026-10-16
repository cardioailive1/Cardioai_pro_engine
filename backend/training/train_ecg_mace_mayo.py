"""
Train the ECG -> MACE networks on real Mayo data, inside the Mayo Workspace.

    python -m training.train_ecg_mace_mayo --rows ecg_rows.csv --dtype int16 --n-leads 12 \\
        --layout lead_major --sample-rate 500 --out weights

Input CSV (one row per ECG, built in the Workspace from the ECG table + OMOP outcomes):
    patient_id, waveform_b64, label_window, [age, sex_male], [heart_rate_bpm, qt_interval_ms, hrv_sdnn_ms]
label_window is one of early_30_90 / none_observed / (imminent_0_30, late_90_365 are dropped).
The waveform layout flags must match what Mayo confirmed (see ecg_waveform.probe_waveform).

For the 12-lead model and the single-lead (wearable proxy) model this:
  1. trains, calibrates on validation, evaluates on held-out patients
  2. fits the feature baseline (logistic regression on the columns given) on the SAME split and runs
     the paired comparison - needed before the checkpoint can ever be approved
  3. writes the checkpoint (NOT approved, tier cutoffs unset) and cutoff_proposals_<variant>.json
Afterwards, humans: pick cutoffs -> set_tier_cutoffs -> approve_checkpoint (ecg_models/checkpoint.py).
Only aggregates (metrics, JSON tables) should leave the Workspace.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ecg_models.checkpoint import save_checkpoint
from ecg_models.compare import paired_auc_delta, propose_tier_cutoffs
from training.mayo.ecg_mace_dataset import FEATURE_COLUMNS, build_examples
from training.train_ecg_mace import train_ecg_mace_net


def _baseline_scores(train, test, cols):
    """Feature baseline on the same split. None if the feature columns are missing."""
    if not cols or any(c not in e["features"] for e in train + test for c in cols):
        return None
    X = lambda ex: np.array([[e["features"][c] for c in cols] for e in ex])
    sc = StandardScaler().fit(X(train))
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(X(train)), [e["y"] for e in train])
    return lr.predict_proba(sc.transform(X(test)))[:, 1]


def run_variant(name, rows, out, *, length, rate, leads, layout, epochs, baseline_cols, wearable_aug):
    ex, skipped = build_examples(rows, length=length, leads=leads, **layout)
    print(f"[{name}] {len(ex)} usable ECGs, {sum(e['y'] for e in ex)} events; skipped {skipped}")
    model, r = train_ecg_mace_net(ex, n_epochs=epochs, wearable_aug=wearable_aug)
    train, val, test = r["splits"]
    print(f"[{name}] data sufficiency: {r['sufficiency']}")
    base = _baseline_scores(train, test, baseline_cols)
    comparison = None
    if base is None:
        print(f"[{name}] WARNING: baseline feature columns missing; no comparison, checkpoint cannot be approved")
    else:
        comparison = paired_auc_delta(r["test_y"], r["test_scores_calibrated"], base)
        print(f"[{name}] network vs baseline: {comparison}")
    props = propose_tier_cutoffs(r["val_y"], r["val_scores_calibrated"], r["test_y"], r["test_scores_calibrated"])
    out = Path(out)
    (out / f"cutoff_proposals_{name}.json").parent.mkdir(parents=True, exist_ok=True)
    (out / f"cutoff_proposals_{name}.json").write_text(json.dumps(props, indent=2))
    fname = "ecg_mace_v1.pt" if leads is None else "ecg_mace_1lead_v1.pt"
    save_checkpoint(model, str(out / fname), sample_rate_hz=rate, length=length, platt=r["platt"],
                    trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes)",
                    metrics={k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r["test"].items() if k in ("auc", "average_precision", "brier")},
                    comparison=comparison, tier_cutoffs=None)
    print(f"[{name}] wrote {out / fname} (not approved) and cutoff proposals:")
    for p in props:
        print("   ", p)
    return comparison, props


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", default="weights")
    ap.add_argument("--dtype", default="int16"); ap.add_argument("--n-leads", type=int, default=12)
    ap.add_argument("--byteorder", default="little"); ap.add_argument("--layout", default="lead_major")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--sample-rate", type=float, required=True, help="Hz, as confirmed by Mayo")
    ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--single-lead-index", type=int, default=0, help="lead for the wearable proxy model; -1 to skip")
    ap.add_argument("--baseline-features", default="heart_rate_bpm,qt_interval_ms", help="comma list; HRV omitted by default (10 s ECG)")
    a = ap.parse_args(argv)
    csv.field_size_limit(sys.maxsize)
    with open(a.rows, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["age"] = float(r["age"]) if r.get("age") else 65.0
        r["sex_male"] = float(r["sex_male"]) if r.get("sex_male") else 0.0
    layout = dict(dtype=a.dtype, n_leads=a.n_leads, byteorder=a.byteorder, layout=a.layout, gain=a.gain)
    cols = [c for c in a.baseline_features.split(",") if c]
    assert all(c in FEATURE_COLUMNS for c in cols), f"baseline features must be in {FEATURE_COLUMNS}"
    length = int(a.sample_rate * a.seconds)
    kw = dict(length=length, rate=a.sample_rate, layout=layout, epochs=a.epochs, baseline_cols=cols)
    run_variant("12lead", rows, a.out, leads=None, wearable_aug=False, **kw)
    if a.single_lead_index >= 0:
        run_variant("1lead", rows, a.out, leads=[a.single_lead_index], wearable_aug=True, **kw)


if __name__ == "__main__":
    main()
