"""
CardioAI Pro — Model Training Script
=========================================
Trains and evaluates a MACE risk model against the label schema in
label_schema.py, specifically on the early_detection_eval_set() — the
cohort that isolates genuine 30-90 day early-detection signal from
imminent-event detection (see label_schema.py's module docstring for why
that distinction is the whole methodological point).

Run it:  python -m training.train_model

WHAT THIS PROVES AND WHAT IT DOESN'T:
Running this against the bundled synthetic generator (synthetic_dataset.py)
proves the pipeline itself works end-to-end — data generation, label
validation, patient-level splitting, training, and evaluation all
function correctly together. It does NOT produce a model suitable for
the live engine, because the "outcomes" are synthetic, not real patient
MACE events. inference/models.py's load_trained_model() must stay
returning None until this same pipeline is run against a real,
IRB-approved, clinically-adjudicated outcomes dataset. Swapping the
data source in main() below is the only change needed to do that —
everything downstream (validation, splitting, training, evaluation)
is real and doesn't change.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

from training.label_schema import LabelValidator, LabelWindow
from training.synthetic_dataset import generate_cohort
from training.benchmark_model import BenchmarkModel


def patient_level_split(records, train_frac=0.7, val_frac=0.15, seed=23):
    """
    Splits by PATIENT, not by record — the same patient's data must never
    appear in both train and test, or evaluation metrics leak information
    and look better than the model actually is.
    """
    rng = np.random.RandomState(seed)
    patient_ids = sorted({r.patient_id for r in records})
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    train_ids = set(patient_ids[: int(n * train_frac)])
    val_ids = set(patient_ids[int(n * train_frac): int(n * (train_frac + val_frac))])
    test_ids = set(patient_ids[int(n * (train_frac + val_frac)):])

    train = [r for r in records if r.patient_id in train_ids]
    val = [r for r in records if r.patient_id in val_ids]
    test = [r for r in records if r.patient_id in test_ids]
    return train, val, test


def labels_for(records) -> np.ndarray:
    return np.array([1 if r.label_window == LabelWindow.EARLY_30_90 else 0 for r in records])


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.3) -> dict:
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"\n--- {name} ---")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Average precision:  {ap:.4f}  (PR-AUC — more informative than ROC-AUC under class imbalance)")
    print(f"  Brier score:        {brier:.4f}  (calibration — lower is better)")
    print(f"  At FIXED threshold {threshold}: sensitivity={sensitivity:.3f}  specificity={specificity:.3f}  PPV={ppv:.3f}")
    print(f"  Confusion matrix:   TP={tp} FP={fp} TN={tn} FN={fn}")
    return {"auc": auc, "ap": ap, "brier": brier, "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv}


def threshold_for_target_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target: float = 0.80) -> float:
    """
    Finds this model's OWN threshold that hits a target sensitivity — the
    same technique fairness/bias_audit.py uses across subgroups, applied
    here across models instead. Two models with different probability
    calibrations are not comparable at the same fixed cutoff (see the
    docstring note in main()); this makes the comparison apples-to-apples
    by matching operating points instead of matching raw thresholds.
    """
    thresholds = sorted(set(y_score.tolist()), reverse=True)
    best_t, best_gap = 0.5, float("inf")
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        gap = abs(sens - target)
        if gap < best_gap:
            best_gap, best_t = gap, t
    return best_t


def evaluate_at_matched_sensitivity(name: str, y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.80) -> dict:
    t = threshold_for_target_sensitivity(y_true, y_score, target_sensitivity)
    y_pred = (y_score >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    print(f"  At MATCHED ~{target_sensitivity:.0%} sensitivity (model's own threshold={t:.4f}): sensitivity={sensitivity:.3f}  specificity={specificity:.3f}  PPV={ppv:.3f}")
    return {"threshold": t, "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv}


def main():
    print("=" * 70)
    print("STEP 1 — Generate synthetic cohort (SYNTHETIC DATA — see module docstring)")
    print("=" * 70)
    records = generate_cohort(n_patients=4000, seed=17)
    print(f"Generated {len(records)} synthetic index observations.")

    print("\n" + "=" * 70)
    print("STEP 2 — Validate against label_schema.py")
    print("=" * 70)
    validator = LabelValidator()
    included, report = validator.validate_dataset(records)
    print(f"Included: {report.included_records}  Excluded: {report.excluded_records}  Reasons: {report.exclusion_reasons}")
    print(f"Window distribution: {report.window_distribution}")
    if report.issues:
        print("Dataset health issues:")
        for issue in report.issues:
            print(f"  - {issue}")

    print("\n" + "=" * 70)
    print("STEP 3 — Build the early-detection eval set (the ONLY methodologically valid cohort for the 30-90 day claim)")
    print("=" * 70)
    eval_set = validator.early_detection_eval_set(included)
    n_pos = sum(1 for r in eval_set if r.label_window == LabelWindow.EARLY_30_90)
    n_neg = sum(1 for r in eval_set if r.label_window == LabelWindow.NONE_OBSERVED)
    print(f"Eval-eligible records: {len(eval_set)}  (positives={n_pos}, negatives={n_neg}, base rate={n_pos/len(eval_set):.3f})")
    print("Excluded from this task entirely: imminent (0-30d) events and late (90-365d) events — see label_schema.py docstring for why.")

    print("\n" + "=" * 70)
    print("STEP 4 — Patient-level train/val/test split")
    print("=" * 70)
    train, val, test = patient_level_split(eval_set)
    print(f"Train: {len(train)} records ({len({r.patient_id for r in train})} patients)")
    print(f"Val:   {len(val)} records ({len({r.patient_id for r in val})} patients)")
    print(f"Test:  {len(test)} records ({len({r.patient_id for r in test})} patients)")
    overlap = {r.patient_id for r in train} & {r.patient_id for r in test}
    print(f"Patient overlap between train and test (must be 0): {len(overlap)}")

    y_train, y_val, y_test = labels_for(train), labels_for(val), labels_for(test)

    print("\n" + "=" * 70)
    print("STEP 5 — Train the benchmark (interpretable logistic regression)")
    print("=" * 70)
    benchmark = BenchmarkModel().fit(train, y_train)
    print("Fitted coefficients (standardized features):", benchmark.coefficients())
    benchmark_test_scores = benchmark.predict_proba(test)
    benchmark_metrics = evaluate("BENCHMARK — Logistic Regression", y_test, benchmark_test_scores)
    benchmark_matched = evaluate_at_matched_sensitivity("BENCHMARK (matched)", y_test, benchmark_test_scores)

    print("\n" + "=" * 70)
    print("STEP 6 — Train the candidate (gradient boosting — can learn the HR x HRV interaction)")
    print("=" * 70)
    X_train = np.array([[r.features[f] for f in BenchmarkModel.FEATURE_NAMES] for r in train])
    X_test = np.array([[r.features[f] for f in BenchmarkModel.FEATURE_NAMES] for r in test])
    candidate = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=23)
    candidate.fit(X_train, y_train)
    candidate_test_scores = candidate.predict_proba(X_test)[:, 1]
    candidate_metrics = evaluate("CANDIDATE — Gradient Boosting", y_test, candidate_test_scores)
    candidate_matched = evaluate_at_matched_sensitivity("CANDIDATE (matched)", y_test, candidate_test_scores)

    print("\n" + "=" * 70)
    print("STEP 7 — Head-to-head comparison")
    print("=" * 70)
    delta_auc = candidate_metrics["auc"] - benchmark_metrics["auc"]
    print(f"AUC delta (candidate - benchmark): {delta_auc:+.4f}")
    print(f"{'Candidate beats benchmark on AUC.' if delta_auc > 0 else 'Benchmark matches or beats the candidate — added complexity is not paying for itself here.'}")
    print(
        "\nNote on the FIXED-threshold numbers above: the two models have very different\n"
        "probability calibrations (see Brier scores), so comparing them at the same raw\n"
        "cutoff (0.3) is misleading — it can make one model look dramatically better or\n"
        "worse purely from calibration, not discrimination. The MATCHED-sensitivity numbers\n"
        "(each model's own threshold for ~80% sensitivity) are the fair comparison:"
    )
    print(f"  Benchmark @ matched: specificity={benchmark_matched['specificity']:.3f}  PPV={benchmark_matched['ppv']:.3f}")
    print(f"  Candidate @ matched: specificity={candidate_matched['specificity']:.3f}  PPV={candidate_matched['ppv']:.3f}")

    print("\n" + "=" * 70)
    print("NOT DONE: wiring either model into inference/models.py")
    print("=" * 70)
    print(
        "Both models above were trained on SYNTHETIC outcomes. Neither is loaded into\n"
        "load_trained_model() and neither should be — that hook stays returning None\n"
        "until this exact pipeline (unchanged from Step 2 onward) is re-run against a\n"
        "real, IRB-approved, clinically-adjudicated MACE outcomes dataset."
    )

    return {"benchmark": benchmark_metrics, "candidate": candidate_metrics}


if __name__ == "__main__":
    main()
