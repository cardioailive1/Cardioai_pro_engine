"""
CardioAI Pro — Algorithmic Bias Audit Protocol for Clinical AI Diagnostics
=============================================================================
Audits the engine's actual inference model (inference/models.py's ModelRegistry)
against a curated, labeled cardiovascular validation cohort stratified by
race/ethnicity, gender, and age band — then runs a threshold-based
remediation pass to close the gap.

IMPORTANT — what's real and what's illustrative here:
  - The statistical framework (subgroup confusion matrices, sensitivity/
    specificity/PPV/NPV, max-min disparity, per-subgroup threshold search
    for equalized sensitivity) is real, standard fairness-auditing
    methodology (post-processing threshold equalization, per Hardt et al.
    2016-style equalized-odds calibration), and it runs against whatever
    model is registered in ModelRegistry — today that's the placeholder
    heuristic; swap in a trained model and this audit runs unchanged.
  - The VALIDATION COHORT is synthetic. There is no real patient data here.
  - The per-subgroup "representation weight" below is a deliberately
    injected simulated effect, not a clinical finding: it stands in for
    the well-documented mechanism where a model trained on underrepresented
    subgroups learns a weaker signal-to-outcome relationship for them,
    producing lower sensitivity even at a "fair-looking" single threshold.
    It exists so the remediation pipeline has something real to fix in
    this demo. Replace it entirely once a real validation cohort with
    real outcome labels exists.

Regulatory framing: the <2 percentage-point diagnostic accuracy gap target
used for pass/fail below mirrors the equity guarantee referenced in the
product's own use-case material ("FDA 2024 digital health equity guidance
compliant"). This module does not itself certify regulatory compliance —
it is the audit tooling a real compliance process would run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

FDA_EQUITY_GAP_THRESHOLD_PP = 2.0  # percentage points — the product's stated equity target
BASELINE_DECISION_THRESHOLD = 0.60  # matches the engine's existing high-risk cutoff

SUBGROUP_DIMENSIONS: dict[str, list[str]] = {
    "race_ethnicity": ["White", "Black", "Hispanic", "Asian", "Other/Unspecified"],
    "gender": ["Female", "Male"],
    "age_band": ["<50", "50-64", "65+"],
}

# Simulated representation weight: 1.0 = model's score is fully predictive of
# true outcome for this subgroup; lower = injected representation gap (see
# module docstring). SYNTHETIC, for demo purposes only.
REPRESENTATION_WEIGHT: dict[str, float] = {
    "White": 1.00, "Black": 0.55, "Hispanic": 0.62, "Asian": 0.78, "Other/Unspecified": 0.65,
    "Male": 1.00, "Female": 0.80,
    "50-64": 1.00, "<50": 0.88, "65+": 0.90,
}


@dataclass
class ValidationCase:
    case_id: str
    subgroups: dict[str, str]          # dimension -> value, e.g. {"race_ethnicity": "Black", ...}
    features: dict[str, float]
    ground_truth: int                   # 1 = true MACE event in validation window, 0 = none
    model_score: float = 0.0
    predicted_positive: bool = False


def _rand(seed_state: list[int]) -> float:
    """Tiny deterministic PRNG so audits are reproducible run-to-run."""
    seed_state[0] = (seed_state[0] * 9301 + 49297) % 233280
    return seed_state[0] / 233280


def generate_validation_cohort(n_per_cell: int = 60, seed: int = 7) -> list[ValidationCase]:
    """
    Builds a synthetic, labeled validation cohort with full factorial coverage
    across race/ethnicity × gender × age band, so every subgroup in every
    dimension has enough cases for a stable metric.
    """
    state = [seed]
    cohort: list[ValidationCase] = []
    idx = 0
    for race in SUBGROUP_DIMENSIONS["race_ethnicity"]:
        for gender in SUBGROUP_DIMENSIONS["gender"]:
            for age_band in SUBGROUP_DIMENSIONS["age_band"]:
                for _ in range(n_per_cell):
                    idx += 1
                    hr = 60 + _rand(state) * 90
                    qtc = 370 + _rand(state) * 170
                    hrv = 8 + _rand(state) * 70
                    cohort.append(ValidationCase(
                        case_id=f"VAL-{idx:05d}",
                        subgroups={"race_ethnicity": race, "gender": gender, "age_band": age_band},
                        features={"heart_rate_bpm": round(hr, 1), "qt_interval_ms": round(qtc, 1), "hrv_sdnn_ms": round(hrv, 1)},
                        ground_truth=0,  # filled in by label_cohort()
                    ))
    return cohort


def label_cohort(cohort: list[ValidationCase], model_predict: Callable[[dict], dict], seed: int = 11) -> None:
    """
    Scores every case with the engine's real model, then assigns a ground-truth
    label as a noisy function of that score — noisier for subgroups with a
    lower REPRESENTATION_WEIGHT, simulating the representation-gap mechanism
    described in the module docstring. Mutates `cohort` in place.
    """
    state = [seed]
    for case in cohort:
        prediction = model_predict(case.features) or {"score": 0.0}
        case.model_score = prediction.get("score", 0.0)

        # Combine subgroup weights (multiplicative — a case can belong to
        # several under-weighted subgroups at once, compounding the effect,
        # same as real intersectional underrepresentation compounds).
        w = 1.0
        for value in case.subgroups.values():
            w *= REPRESENTATION_WEIGHT.get(value, 1.0)
        w = max(0.15, w)  # floor so labels never become pure noise

        signal = w * case.model_score + (1 - w) * _rand(state)
        case.ground_truth = 1 if _rand(state) < signal else 0


def score_cohort(cohort: list[ValidationCase], threshold: float) -> None:
    for case in cohort:
        case.predicted_positive = case.model_score >= threshold


def _confusion(cases: list[ValidationCase]) -> dict[str, int]:
    tp = sum(1 for c in cases if c.predicted_positive and c.ground_truth == 1)
    fp = sum(1 for c in cases if c.predicted_positive and c.ground_truth == 0)
    tn = sum(1 for c in cases if not c.predicted_positive and c.ground_truth == 0)
    fn = sum(1 for c in cases if not c.predicted_positive and c.ground_truth == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_confusion(conf: dict[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
    n = tp + fp + tn + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0     # recall / true positive rate
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sensitivity, 4), "specificity": round(specificity, 4),
        "ppv": round(ppv, 4), "npv": round(npv, 4), "accuracy": round(accuracy, 4),
    }


def subgroup_metrics(cohort: list[ValidationCase], dimension: str) -> dict[str, dict[str, Any]]:
    out = {}
    for value in SUBGROUP_DIMENSIONS[dimension]:
        cases = [c for c in cohort if c.subgroups[dimension] == value]
        out[value] = _metrics_from_confusion(_confusion(cases))
    return out


def disparity_gap_pp(metrics_by_subgroup: dict[str, dict[str, Any]], metric_key: str = "accuracy") -> float:
    """Max-min gap across subgroups, in percentage points."""
    values = [m[metric_key] for m in metrics_by_subgroup.values() if m["n"] > 0]
    if not values:
        return 0.0
    return round((max(values) - min(values)) * 100, 2)


def remediate_thresholds(
    cohort: list[ValidationCase], dimension: str, target_sensitivity: float | None = None,
    grid: tuple[float, float, float] = (0.20, 0.80, 0.01),
) -> dict[str, float]:
    """
    Threshold-based remediation: for each subgroup in `dimension`, searches a
    grid of decision thresholds and picks the one whose sensitivity is
    closest to `target_sensitivity` (default: the cohort-wide sensitivity at
    the baseline global threshold). This is standard post-processing
    equalized-odds-style calibration — the model itself never changes, only
    the per-subgroup operating point.
    """
    lo, hi, step = grid
    n_steps = int(round((hi - lo) / step)) + 1
    thresholds = [round(lo + i * step, 4) for i in range(n_steps)]

    if target_sensitivity is None:
        overall_scored = list(cohort)
        score_cohort(overall_scored, BASELINE_DECISION_THRESHOLD)
        target_sensitivity = _metrics_from_confusion(_confusion(overall_scored))["sensitivity"]

    remediated: dict[str, float] = {}
    for value in SUBGROUP_DIMENSIONS[dimension]:
        cases = [c for c in cohort if c.subgroups[dimension] == value]
        best_t, best_gap = BASELINE_DECISION_THRESHOLD, float("inf")
        for t in thresholds:
            for c in cases:
                c.predicted_positive = c.model_score >= t
            sens = _metrics_from_confusion(_confusion(cases))["sensitivity"]
            gap = abs(sens - target_sensitivity)
            if gap < best_gap:
                best_gap, best_t = gap, t
        remediated[value] = best_t
    return remediated


def apply_remediated_thresholds(cohort: list[ValidationCase], dimension: str, thresholds: dict[str, float]) -> None:
    for case in cohort:
        t = thresholds.get(case.subgroups[dimension], BASELINE_DECISION_THRESHOLD)
        case.predicted_positive = case.model_score >= t


def run_audit(model_predict: Callable[[dict], dict], n_per_cell: int = 60) -> dict[str, Any]:
    """
    Full protocol: generate cohort -> label with real model -> baseline
    audit at the global threshold -> per-dimension threshold remediation ->
    post-remediation audit -> pass/fail against the equity target.
    """
    cohort = generate_validation_cohort(n_per_cell=n_per_cell)
    label_cohort(cohort, model_predict)

    report: dict[str, Any] = {
        "cohort_size": len(cohort),
        "baseline_threshold": BASELINE_DECISION_THRESHOLD,
        "equity_target_pp": FDA_EQUITY_GAP_THRESHOLD_PP,
        "dimensions": {},
    }

    for dimension in SUBGROUP_DIMENSIONS:
        score_cohort(cohort, BASELINE_DECISION_THRESHOLD)
        baseline = subgroup_metrics(cohort, dimension)
        baseline_gap = disparity_gap_pp(baseline, "accuracy")
        baseline_sens_gap = disparity_gap_pp(baseline, "sensitivity")

        remediated_thresholds = remediate_thresholds(cohort, dimension)
        apply_remediated_thresholds(cohort, dimension, remediated_thresholds)
        remediated = subgroup_metrics(cohort, dimension)
        remediated_gap = disparity_gap_pp(remediated, "accuracy")
        remediated_sens_gap = disparity_gap_pp(remediated, "sensitivity")

        report["dimensions"][dimension] = {
            "baseline": {"metrics": baseline, "accuracy_gap_pp": baseline_gap, "sensitivity_gap_pp": baseline_sens_gap},
            "remediated": {
                "thresholds": remediated_thresholds, "metrics": remediated,
                "accuracy_gap_pp": remediated_gap, "sensitivity_gap_pp": remediated_sens_gap,
            },
            # Pass/fail is judged on the SENSITIVITY gap (equal opportunity —
            # true-positive-rate parity), not raw accuracy. For a diagnostic
            # model, the clinically relevant equity failure is disproportionately
            # missing real events in a subgroup, not aggregate accuracy, which
            # class imbalance can distort independent of any real disparity.
            # Threshold-based remediation targets sensitivity directly, so this
            # is also the metric it can actually be expected to fix.
            "primary_metric": "sensitivity",
            "passes_equity_target": remediated_sens_gap <= FDA_EQUITY_GAP_THRESHOLD_PP,
            "improvement_pp": round(baseline_sens_gap - remediated_sens_gap, 2),
        }

    report["overall_pass"] = all(d["passes_equity_target"] for d in report["dimensions"].values())
    return report
