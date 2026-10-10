"""
CardioAI Pro — Synthetic Training Dataset Generator
========================================================
Generates a MACELabelRecord-compliant synthetic cohort so the training
pipeline (train_model.py) can be run and tested end-to-end. This is NOT
real patient data and the resulting trained model is NOT suitable for the
live engine — see train_model.py's closing note for why.

The synthetic hazard function deliberately includes a nonlinear
interaction (high heart rate AND low HRV together is worse than either
alone, roughly modeling real autonomic-dysfunction physiology) so a
linear benchmark and a nonlinear candidate model are meaningfully
different in how well they can learn it — the same reason real MACE risk
modeling looks at feature interactions, not just univariate thresholds.

Representation weighting reuses the same mechanism as
fairness/bias_audit.py, so a model trained on this synthetic cohort can
be run through the existing bias audit for a coherent end-to-end story.
"""
from __future__ import annotations

import random
from typing import Any

from training.label_schema import (
    MACELabelRecord, MACEEventType, SUBGROUP_DIMENSIONS, FEATURE_RANGES,
)

REPRESENTATION_WEIGHT: dict[str, float] = {
    "White": 1.00, "Black": 0.55, "Hispanic": 0.62, "Asian": 0.78, "Other/Unspecified": 0.65,
    "Male": 1.00, "Female": 0.80,
    "50-64": 1.00, "<50": 0.88, "65+": 0.90,
}

MAX_FOLLOWUP_DAYS = 365


def _true_hazard(hr: float, qtc: float, hrv: float) -> float:
    """
    Synthetic ground-truth risk function — NOT a clinical formula, just
    something with a genuine nonlinear interaction for the benchmark vs.
    candidate model comparison to be meaningful. Roughly: elevated HR and
    depressed HRV are each mildly informative alone, but their *combination*
    is disproportionately worse — modeling autonomic dysfunction, where
    it's the pairing that's the real red flag, not either alone.
    """
    hr_term = max(0.0, (hr - 85) / 60)
    qtc_term = max(0.0, (qtc - 430) / 100)
    hrv_term = max(0.0, (45 - hrv) / 40)
    interaction = hr_term * hrv_term * 1.8  # the nonlinear part a linear model can't easily capture
    hazard = 0.15 * hr_term + 0.15 * qtc_term + 0.15 * hrv_term + interaction
    return max(0.0, min(hazard, 3.0))


def generate_cohort(n_patients: int = 4000, seed: int = 17) -> list[MACELabelRecord]:
    rnd = random.Random(seed)
    records: list[MACELabelRecord] = []

    for i in range(n_patients):
        patient_id = f"SYN-{i:05d}"
        race = rnd.choice(SUBGROUP_DIMENSIONS["race_ethnicity"])
        gender = rnd.choice(SUBGROUP_DIMENSIONS["gender"])
        age_band = rnd.choice(SUBGROUP_DIMENSIONS["age_band"])

        lo_hr, hi_hr = FEATURE_RANGES["ecg"]["heart_rate_bpm"]
        lo_qtc, hi_qtc = FEATURE_RANGES["ecg"]["qt_interval_ms"]
        lo_hrv, hi_hrv = FEATURE_RANGES["ecg"]["hrv_sdnn_ms"]
        # Sampled within a clinically plausible sub-range of each feature's
        # full valid range (not the full range — most real readings cluster,
        # they don't span the whole physiologically-possible envelope).
        hr = rnd.uniform(max(lo_hr, 55), min(hi_hr, 150))
        qtc = rnd.uniform(max(lo_qtc, 370), min(hi_qtc, 520))
        hrv = rnd.uniform(max(lo_hrv, 10), min(hi_hrv, 80))

        hazard = _true_hazard(hr, qtc, hrv)

        # Representation gap: for under-weighted subgroups, blend the true
        # hazard with noise, same mechanism as the bias audit — the
        # feature-outcome relationship is genuinely weaker for these
        # groups in the synthetic data, standing in for real-world
        # under-representation in training data.
        w = REPRESENTATION_WEIGHT.get(race, 1.0) * REPRESENTATION_WEIGHT.get(gender, 1.0) * REPRESENTATION_WEIGHT.get(age_band, 1.0)
        w = max(0.2, w)
        observed_hazard = w * hazard + (1 - w) * rnd.uniform(0, 1.0)

        # Convert hazard to a time-to-event via an exponential-ish draw —
        # higher hazard means shorter expected time to event.
        event_prob_in_window = 1 - pow(2.71828, -observed_hazard)
        has_event = rnd.random() < min(event_prob_in_window, 0.85)

        if has_event:
            # Higher hazard -> earlier events, via inverse-hazard-scaled draw
            scale = max(10, 180 / (observed_hazard + 0.15))
            days_to_event = int(min(MAX_FOLLOWUP_DAYS, rnd.expovariate(1 / scale) + 1))
            event_type = rnd.choice([MACEEventType.MYOCARDIAL_INFARCTION, MACEEventType.CARDIOVASCULAR_DEATH, MACEEventType.STROKE])
            follow_up_days = days_to_event
            censored = False
        else:
            days_to_event = None
            event_type = None
            follow_up_days = MAX_FOLLOWUP_DAYS
            censored = True

        # Washout: ~4% of records simulate a recent prior MACE (excluded downstream)
        days_since_prior_mace = rnd.randint(0, 60) if rnd.random() < 0.04 else None

        records.append(MACELabelRecord(
            record_id=f"REC-{i:05d}", patient_id=patient_id, modality="ecg", index_timestamp=0.0,
            features={"heart_rate_bpm": round(hr, 1), "qt_interval_ms": round(qtc, 1), "hrv_sdnn_ms": round(hrv, 1)},
            event_type=event_type, days_to_event=days_to_event, follow_up_days=follow_up_days, censored=censored,
            days_since_prior_mace=days_since_prior_mace,
            subgroups={"race_ethnicity": race, "gender": gender, "age_band": age_band},
        ))

    return records
