"""
CardioAI Pro — MACE Training Label Specification
=====================================================
Defines the labels a real MACE risk model would need to be trained and
validated against, specifically so that "detects MACE risk 30-90 days
early" is a claim the model has actually been shown to be true of — not
a hardcoded window_days field sitting in a dictionary (see the note in
inference/models.py; that's exactly the gap this closes).

THE CENTRAL METHODOLOGICAL POINT — read this before using the schema:
A model that's simply good at detecting CURRENT acute abnormality will
trivially look good at "predicting" events happening in the next few
days, because acute abnormality and imminent events are correlated for
mundane reasons — the patient is already visibly sick. That is NOT early
detection, and a training/evaluation setup that doesn't guard against it
will produce a model that silently cheats: it "predicts" 30-90 days out
by actually just detecting events about to happen in the next 0-30 days
whenever those events are preceded by 30-90 day-old warning signs.

The guard is exclusion, not just labeling: any index observation with an
event in the [0, 30) day window is excluded from the 30-90 day training
and evaluation set entirely (see EARLY_ONLY_EXCLUSIONS below). The model
is only ever scored, on the 30-90 day task, against cases where NOTHING
was happening in the first 30 days — so a positive prediction can only be
explained by a genuine early signal, not by detecting an already-unfolding
event.

Every feature key below matches ingestion/quality.py's SCHEMAS exactly and
every subgroup value matches fairness/bias_audit.py's SUBGROUP_DIMENSIONS
exactly, so a real labeled dataset built to this spec can be run through
this system's existing quality checks and bias audit machinery unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Outcome definition
# ---------------------------------------------------------------------------
class MACEEventType(str, Enum):
    """
    3-point MACE by default (cardiovascular death, non-fatal MI, non-fatal
    stroke) is the most common composite in cardiology literature and is
    what MACE_QUALIFYING below uses. Revascularization and unstable-angina
    hospitalization are included as *recorded* event types (useful for
    subgroup analysis and for teams that want a 4- or 5-point composite)
    but are NOT counted toward the default composite label — swap
    MACE_QUALIFYING if your clinical team's definition differs; this is a
    real point of variation across studies and should be a deliberate
    choice, not an accident of this file's defaults.
    """
    CARDIOVASCULAR_DEATH = "cardiovascular_death"
    MYOCARDIAL_INFARCTION = "myocardial_infarction"
    STROKE = "stroke"
    REVASCULARIZATION = "revascularization"
    UNSTABLE_ANGINA_HOSPITALIZATION = "unstable_angina_hospitalization"


MACE_QUALIFYING: frozenset[MACEEventType] = frozenset({
    MACEEventType.CARDIOVASCULAR_DEATH,
    MACEEventType.MYOCARDIAL_INFARCTION,
    MACEEventType.STROKE,
})


class LabelWindow(str, Enum):
    """Which forward-looking window, relative to the index observation, the event fell into."""
    IMMINENT_0_30 = "imminent_0_30"      # excluded from the 30-90d task — see module docstring
    EARLY_30_90 = "early_30_90"          # the window the marketing claim is actually about
    LATE_90_365 = "late_90_365"          # negative control — event happened, but too late to count as "early"
    NONE_OBSERVED = "none_observed"      # no qualifying event in the follow-up window (censored or true negative)


# ---------------------------------------------------------------------------
# Feature schema — kept identical to ingestion/quality.py's SCHEMAS so a
# labeled training row and a live ingested record are structurally the
# same shape.
# ---------------------------------------------------------------------------
FEATURE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "ecg": {"heart_rate_bpm": (30, 220), "qt_interval_ms": (250, 550), "hrv_sdnn_ms": (5, 200)},
    "wearable": {"heart_rate_bpm": (30, 220), "spo2_pct": (70, 100), "steps": (0, 60000)},
}

# Kept identical to fairness/bias_audit.py's SUBGROUP_DIMENSIONS so a real
# labeled dataset can be run through the existing bias audit unchanged.
SUBGROUP_DIMENSIONS: dict[str, list[str]] = {
    "race_ethnicity": ["White", "Black", "Hispanic", "Asian", "Other/Unspecified"],
    "gender": ["Female", "Male"],
    "age_band": ["<50", "50-64", "65+"],
}

MIN_FOLLOWUP_DAYS_FOR_NEGATIVE = 90   # can't confirm "no event" without watching at least this long
WASHOUT_DAYS_BEFORE_INDEX = 90        # exclude if a qualifying event occurred in the 90 days before index


@dataclass
class MACELabelRecord:
    """One training row: an index observation, its features, and its forward-looking outcome."""
    record_id: str
    patient_id: str
    modality: str                              # "ecg" | "wearable" — which feature set this row carries
    index_timestamp: float                      # "time zero" — when the observation was taken
    features: dict[str, float]

    # Outcome / follow-up
    event_type: Optional[MACEEventType]          # None if no event observed
    days_to_event: Optional[int]                  # None if censored (no event within follow-up)
    follow_up_days: int                            # how long this patient was actually observed after index
    censored: bool                                  # True if no event was observed within follow_up_days

    # Prior-event context, needed for washout exclusion
    days_since_prior_mace: Optional[int] = None      # None if no prior MACE on record

    # Subgroup labels, for bias-audit compatibility
    subgroups: dict[str, str] = field(default_factory=dict)

    # Filled in by compute_window() / validate() — not set by hand
    label_window: Optional[LabelWindow] = None
    exclude: bool = False
    exclude_reasons: list[str] = field(default_factory=list)

    def is_mace_qualifying(self) -> bool:
        return self.event_type is not None and self.event_type in MACE_QUALIFYING


def compute_window(record: MACELabelRecord) -> LabelWindow:
    """Determines which forward-looking window a record's outcome falls into."""
    if record.censored or not record.is_mace_qualifying():
        return LabelWindow.NONE_OBSERVED
    d = record.days_to_event
    if d is None:
        return LabelWindow.NONE_OBSERVED
    if d < 30:
        return LabelWindow.IMMINENT_0_30
    if d < 90:
        return LabelWindow.EARLY_30_90
    return LabelWindow.LATE_90_365


@dataclass
class ValidationReport:
    total_records: int
    included_records: int
    excluded_records: int
    exclusion_reasons: dict[str, int]
    window_distribution: dict[str, int]
    subgroup_coverage: dict[str, dict[str, int]]
    feature_range_violations: dict[str, int]
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.included_records > 0 and not any(
            v > 0 for v in self.feature_range_violations.values()
        )


class LabelValidator:
    """
    Applies exclusion criteria, computes label windows, and reports dataset
    health — class balance, subgroup coverage, feature range compliance —
    before a dataset is handed to a training run. This does not train
    anything; it's the gate a real dataset needs to pass before it's safe
    to train on.
    """

    def validate_record(self, record: MACELabelRecord) -> MACELabelRecord:
        reasons: list[str] = []

        # Washout: a qualifying event shortly before index means this
        # observation is in the middle of an existing event, not a clean
        # "time zero" for predicting a NEW one.
        if record.days_since_prior_mace is not None and record.days_since_prior_mace < WASHOUT_DAYS_BEFORE_INDEX:
            reasons.append("prior_mace_within_washout")

        # Insufficient follow-up: can't confirm a negative without watching
        # long enough. A censored record with only 20 days of follow-up
        # tells you nothing about whether an event happened on day 45.
        if record.censored and record.follow_up_days < MIN_FOLLOWUP_DAYS_FOR_NEGATIVE:
            reasons.append("insufficient_followup_for_negative")

        # Feature range compliance — reuses the same bounds the live
        # ingestion pipeline enforces, so a labeled row that wouldn't have
        # passed DataQualityEngine in production doesn't get trained on.
        ranges = FEATURE_RANGES.get(record.modality, {})
        for feat, (lo, hi) in ranges.items():
            val = record.features.get(feat)
            if val is not None and not (lo <= val <= hi):
                reasons.append(f"feature_out_of_range:{feat}")

        # Subgroup completeness — required for the bias audit this dataset
        # is meant to eventually replace the synthetic cohort with.
        missing_subgroups = [dim for dim in SUBGROUP_DIMENSIONS if dim not in record.subgroups]
        if missing_subgroups:
            reasons.append(f"missing_subgroups:{','.join(missing_subgroups)}")

        record.label_window = compute_window(record)
        record.exclude = len(reasons) > 0
        record.exclude_reasons = reasons
        return record

    def validate_dataset(self, records: list[MACELabelRecord]) -> tuple[list[MACELabelRecord], ValidationReport]:
        validated = [self.validate_record(r) for r in records]
        included = [r for r in validated if not r.exclude]
        excluded = [r for r in validated if r.exclude]

        exclusion_reasons: dict[str, int] = {}
        for r in excluded:
            for reason in r.exclude_reasons:
                key = reason.split(":")[0]
                exclusion_reasons[key] = exclusion_reasons.get(key, 0) + 1

        window_distribution: dict[str, int] = {}
        for r in included:
            key = r.label_window.value if r.label_window else "unknown"
            window_distribution[key] = window_distribution.get(key, 0) + 1

        subgroup_coverage: dict[str, dict[str, int]] = {dim: {} for dim in SUBGROUP_DIMENSIONS}
        for r in included:
            for dim, values in SUBGROUP_DIMENSIONS.items():
                v = r.subgroups.get(dim)
                if v:
                    subgroup_coverage[dim][v] = subgroup_coverage[dim].get(v, 0) + 1

        range_violations: dict[str, int] = {}
        for r in validated:
            for reason in r.exclude_reasons:
                if reason.startswith("feature_out_of_range"):
                    feat = reason.split(":")[1]
                    range_violations[feat] = range_violations.get(feat, 0) + 1

        issues: list[str] = []
        early_n = window_distribution.get(LabelWindow.EARLY_30_90.value, 0)
        if early_n < 20:
            issues.append(f"Only {early_n} positive examples in the 30-90 day window — likely too few to train a reliable classifier for the specific claim being validated.")
        for dim, values in subgroup_coverage.items():
            thin = [v for v, n in values.items() if n < 10]
            if thin:
                issues.append(f"Thin subgroup coverage in {dim}: {thin} — the bias audit will be unreliable for these.")

        report = ValidationReport(
            total_records=len(records), included_records=len(included), excluded_records=len(excluded),
            exclusion_reasons=exclusion_reasons, window_distribution=window_distribution,
            subgroup_coverage=subgroup_coverage, feature_range_violations=range_violations, issues=issues,
        )
        return included, report

    def early_detection_eval_set(self, included: list[MACELabelRecord]) -> list[MACELabelRecord]:
        """
        The specific evaluation cohort for the "30-90 days early" claim:
        positives are EARLY_30_90 only, negatives are NONE_OBSERVED only.
        IMMINENT_0_30 and LATE_90_365 records are excluded from this
        specific evaluation — mixing them in either direction would let a
        model pass by detecting the wrong thing (see module docstring).
        """
        return [r for r in included if r.label_window in (LabelWindow.EARLY_30_90, LabelWindow.NONE_OBSERVED)]
