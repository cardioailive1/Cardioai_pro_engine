"""
Train the ECG -> MACE networks on real Mayo data, inside the Mayo Workspace.

    python -m training.train_ecg_mace_mayo --rows ecg_rows.csv --dtype int16 --n-leads 12 \\
        --layout lead_major --sample-rate 500 --out weights

Input CSV (one row per ECG, built in the Workspace from the ECG table + OMOP outcomes):
    patient_id, waveform_b64, label_window, [event_type], [age, sex_male],
    [heart_rate_bpm, qt_interval_ms, hrv_sdnn_ms]
For Mayo's FACT_ECG_WAVEFORM use --mayo-leads: replace waveform_b64 with
    leads_json (a JSON object {"I": <WAVEFORM>, "II": ..., "V6": ...} for the 'Rhythm' recording)
    and sample_rate_hz (ECGSAMPLEBASE). --sample-rate is then the common rate every ECG is resampled to
    (500 keeps the 5,000-sample input). patient_id should be PATIENT_CLINIC_NUMBER.
label_window is one of early_30_90 / none_observed / (imminent_0_30, late_90_365 are dropped).
event_type is the MACEEventType value (all_cause_death / myocardial_infarction / stroke / ...) for
positives, empty or absent for negatives. Only needed when --outcome-stratify is set.
The waveform layout flags must match what Mayo confirmed (see ecg_waveform.probe_waveform).

For the 12-lead model and the single-lead (wearable proxy) model this:
  1. trains, calibrates on validation, evaluates on held-out patients
  2. fits the feature baseline (logistic regression on the columns given) on the SAME split and runs
     the paired comparison - needed before the checkpoint can ever be approved
  3. writes the checkpoint (NOT approved, tier cutoffs unset) and cutoff_proposals_<variant>.json
Afterwards, humans: pick cutoffs -> set_tier_cutoffs -> approve_checkpoint (ecg_models/checkpoint.py).
Only aggregates (metrics, JSON tables) should leave the Workspace.

--outcome-stratify trains three variants on the SAME patient split:
  composite   all MACE events (the approvable checkpoint, same as the default run)
  deaths      positives are all_cause_death only; MI/stroke positives become negatives for this variant
  nonfatal    positives are myocardial_infarction or stroke only; deaths become negatives
The composite variant is the one that writes ecg_mace_v1.pt / ecg_mace_1lead_v1.pt and can be
approved for live scoring. The deaths and nonfatal variants write diagnostic-only checkpoints
(ecg_mace_deaths_v1.pt, ecg_mace_nonfatal_v1.pt) with approved_for_live always False — they are
research outputs that inform clinical interpretation but do not replace the composite model.
The patient split is shared across all three so AUC differences are real, not sampling noise.
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
from training.train_ecg_mace import split_by_patient, train_ecg_mace_net

# MACEEventType values that count as deaths / non-fatal for stratified runs.
_DEATH_TYPES = {"all_cause_death", "cardiovascular_death"}
_NONFATAL_TYPES = {"myocardial_infarction", "stroke"}


def _baseline_scores(train, test, cols):
    """Feature baseline on the same split. None if the feature columns are missing."""
    if not cols or any(c not in e["features"] for e in train + test for c in cols):
        return None
    X = lambda ex: np.array([[e["features"][c] for c in cols] for e in ex])
    sc = StandardScaler().fit(X(train))
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(X(train)), [e["y"] for e in train])
    return lr.predict_proba(sc.transform(X(test)))[:, 1]


def _relabel(examples, keep_event_types):
    """Return a copy of examples where positives whose event_type is NOT in
    keep_event_types are flipped to negative (y=0). Negatives are unchanged.
    Used for the deaths-only and nonfatal-only stratified variants."""
    out = []
    for e in examples:
        et = e.get("event_type") or ""
        if e["y"] == 1 and et not in keep_event_types:
            e = dict(e, y=0)   # was a positive for the other outcome type; treat as negative here
        out.append(e)
    return out


def run_variant(name, examples, out, *, length, rate, leads, epochs, baseline_cols,
                wearable_aug, approvable=True, trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes)",
                shared_split=None):
    """Train one network variant.

    shared_split: (train_ids, val_ids, test_ids) sets of patient_id — when provided the split
    is reused so outcome-stratified variants are evaluated on the exact same patients, making
    AUC differences interpretable as signal differences, not sampling noise.
    """
    # Apply lead selection before splitting so we don't duplicate waveform data.
    ex = [dict(e, x=e["x"][list(leads)] if leads is not None else e["x"]) for e in examples]
    n_pos = sum(e["y"] for e in ex)
    print(f"[{name}] {len(ex)} ECGs, {n_pos} positives ({len(ex) - n_pos} negatives)")
    model, r = train_ecg_mace_net(ex, n_epochs=epochs, wearable_aug=wearable_aug,
                                   shared_split=shared_split)
    train, val, test = r["splits"]
    print(f"[{name}] data sufficiency: {r['sufficiency']}")
    base = _baseline_scores(train, test, baseline_cols)
    comparison = None
    if base is None:
        print(f"[{name}] WARNING: baseline feature columns missing; no comparison"
              + ("" if not approvable else "; checkpoint CANNOT be approved"))
    else:
        comparison = paired_auc_delta(r["test_y"], r["test_scores_calibrated"], base)
        print(f"[{name}] network vs baseline: {comparison}")
    props = propose_tier_cutoffs(r["val_y"], r["val_scores_calibrated"], r["test_y"], r["test_scores_calibrated"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cutoff_proposals_{name}.json").write_text(json.dumps(props, indent=2))
    if approvable:
        # Composite model: writes the canonical filename the live engine looks for.
        fname = "ecg_mace_v1.pt" if leads is None else "ecg_mace_1lead_v1.pt"
    else:
        # Stratified diagnostic variants: distinct names, never approvable for live scoring.
        suffix = name.split("_")[0]   # "deaths" or "nonfatal"
        fname = f"ecg_mace_{suffix}_v1.pt" if leads is None else f"ecg_mace_{suffix}_1lead_v1.pt"
    save_checkpoint(model, str(out / fname), sample_rate_hz=rate, length=length, platt=r["platt"],
                    trained_on=trained_on,
                    metrics={k: float(v) if isinstance(v, (int, float, np.floating)) else v
                             for k, v in r["test"].items() if k in ("auc", "average_precision", "brier")},
                    comparison=comparison, tier_cutoffs=None,
                    approvable=approvable)
    print(f"[{name}] wrote {out / fname}"
          + (" (not approved; cutoffs unset)" if approvable else " (diagnostic only; not approvable for live)")
          + " and cutoff proposals:")
    for p in props:
        print("   ", p)
    # Return the patient split so stratified variants can reuse it.
    split_ids = (
        {e["patient_id"] for e in train},
        {e["patient_id"] for e in val},
        {e["patient_id"] for e in test},
    )
    return comparison, props, split_ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", default="weights")
    ap.add_argument("--dtype", default="int16"); ap.add_argument("--n-leads", type=int, default=12)
    ap.add_argument("--byteorder", default="little"); ap.add_argument("--layout", default="lead_major")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--mayo-leads", action="store_true",
                    help="rows carry leads_json + sample_rate_hz (Mayo FACT_ECG_WAVEFORM)")
    ap.add_argument("--sample-rate", type=float, required=True, help="Hz, as confirmed by Mayo")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--single-lead-index", type=int, default=0,
                    help="lead index for the wearable proxy model; -1 to skip")
    ap.add_argument("--baseline-features", default="heart_rate_bpm,qt_interval_ms",
                    help="comma list of feature columns for the logistic baseline; HRV omitted by default")
    ap.add_argument("--outcome-stratify", action="store_true",
                    help="also train deaths-only and nonfatal-only variants on the same patient split. "
                         "Requires event_type column in the CSV. The composite model is still the one "
                         "written to ecg_mace_v1.pt and eligible for approval; the stratified variants "
                         "are research outputs only (approved_for_live always False).")
    a = ap.parse_args(argv)
    csv.field_size_limit(sys.maxsize)
    with open(a.rows, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["age"] = float(r["age"]) if r.get("age") else 65.0
        r["sex_male"] = float(r["sex_male"]) if r.get("sex_male") else 0.0
        if a.mayo_leads:
            r["leads"] = json.loads(r["leads_json"])
            r["sample_rate_hz"] = float(r["sample_rate_hz"]) if r.get("sample_rate_hz") else None

    if a.outcome_stratify and not any(r.get("event_type") for r in rows):
        raise ValueError(
            "--outcome-stratify requires an event_type column in the CSV. "
            "Add it from mace_label_rules: the value is the MACEEventType string for positives, "
            "empty for negatives."
        )

    layout = dict(dtype=a.dtype, n_leads=a.n_leads, byteorder=a.byteorder, layout=a.layout, gain=a.gain)
    cols = [c for c in a.baseline_features.split(",") if c]
    assert all(c in FEATURE_COLUMNS for c in cols), f"baseline features must be in {FEATURE_COLUMNS}"
    length = int(a.sample_rate * a.seconds)

    # Build examples once; lead selection happens inside run_variant.
    ex_all, skipped = build_examples(rows, length=length, leads=None, target_rate_hz=a.sample_rate, **layout)
    print(f"Built {len(ex_all)} usable ECGs; skipped {skipped}")

    # Propagate event_type from the raw rows into examples (build_examples doesn't carry it).
    # Build a lookup: patient_id -> event_type (last non-empty wins if a patient has multiple ECGs).
    pat_event: dict[str, str] = {}
    for r in rows:
        if r.get("event_type"):
            pat_event[r["patient_id"]] = r["event_type"]
    for e in ex_all:
        e["event_type"] = pat_event.get(e["patient_id"], "")

    kw = dict(rate=a.sample_rate, epochs=a.epochs, baseline_cols=cols)

    # --- composite (approvable) ---
    _, _, split_ids = run_variant(
        "12lead", ex_all, a.out, length=length, leads=None, wearable_aug=False,
        approvable=True, **kw,
    )
    if a.single_lead_index >= 0:
        run_variant(
            "1lead", ex_all, a.out, length=length, leads=[a.single_lead_index], wearable_aug=True,
            approvable=True, shared_split=split_ids, **kw,
        )

    if a.outcome_stratify:
        print("\n--- outcome-stratified variants (diagnostic only; not approvable for live scoring) ---")
        # deaths-only: positives = all_cause_death / cardiovascular_death; MI/stroke -> negative
        ex_deaths = _relabel(ex_all, _DEATH_TYPES)
        n_d = sum(e["y"] for e in ex_deaths)
        print(f"deaths variant: {n_d} positives after relabeling")
        if n_d < 30:
            print("WARNING: fewer than 30 death positives — results will be unstable; interpret with caution")
        run_variant(
            "deaths_12lead", ex_deaths, a.out, length=length, leads=None, wearable_aug=False,
            approvable=False, shared_split=split_ids,
            trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes) — deaths-only variant", **kw,
        )
        if a.single_lead_index >= 0:
            run_variant(
                "deaths_1lead", ex_deaths, a.out, length=length, leads=[a.single_lead_index], wearable_aug=True,
                approvable=False, shared_split=split_ids,
                trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes) — deaths-only variant", **kw,
            )

        # nonfatal-only: positives = MI or stroke; deaths -> negative
        ex_nonfatal = _relabel(ex_all, _NONFATAL_TYPES)
        n_nf = sum(e["y"] for e in ex_nonfatal)
        print(f"nonfatal variant: {n_nf} positives after relabeling")
        if n_nf < 30:
            print("WARNING: fewer than 30 non-fatal positives — results will be unstable; interpret with caution")
        run_variant(
            "nonfatal_12lead", ex_nonfatal, a.out, length=length, leads=None, wearable_aug=False,
            approvable=False, shared_split=split_ids,
            trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes) — nonfatal MI/stroke variant", **kw,
        )
        if a.single_lead_index >= 0:
            run_variant(
                "nonfatal_1lead", ex_nonfatal, a.out, length=length, leads=[a.single_lead_index], wearable_aug=True,
                approvable=False, shared_split=split_ids,
                trained_on="Mayo Clinic Platform (real ECG + OMOP outcomes) — nonfatal MI/stroke variant", **kw,
            )
        print("\nStratified summary written to cutoff_proposals_deaths_*.json and cutoff_proposals_nonfatal_*.json")
        print("Only the composite ecg_mace_v1.pt / ecg_mace_1lead_v1.pt are eligible for approve_checkpoint.")


if __name__ == "__main__":
    main()
