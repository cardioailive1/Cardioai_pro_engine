"""
CardioAI Pro — OMOP CDM to MACELabelRecord Adapter
=========================================================
Bridges Mayo Clinic Platform_Discover's OMOP CDM-structured de-identified
data (queried via SparkSQL in a Workspace Jupyter notebook) to
training/label_schema.py's MACELabelRecord — the one piece of genuinely
new code the Mayo training path needs. Everything downstream
(LabelValidator, train_model.py) is unchanged; this module's whole job is
producing MACELabelRecord objects those already handle correctly.

Expects rows shaped like OMOP CDM's standard tables — PERSON,
MEASUREMENT, CONDITION_OCCURRENCE, DEATH — as plain dicts (what you get
from a collected Spark DataFrame or a pandas DataFrame's
`.to_dict("records")`; this module doesn't depend on either library
directly, so it works with whichever your Workspace notebook uses).

IMPORTANT — concept_id VALUES ARE NOT HARDCODED HERE, ON PURPOSE:
OMOP concept_ids for specific conditions and measurements are vocabulary-
version and institution-specific. Hardcoding numbers from memory risks
silently mislabeling real patient data with a subtly WRONG code — a far
worse failure mode than a crash, since it wouldn't be obvious anything
was wrong. Every concept_id this adapter needs is a required field on
OMOPConceptMapping below, filled in by YOU after looking them up in your
own Mayo Discover instance's CONCEPT and CONCEPT_RELATIONSHIP tables
(Schema Visualizer's search tool is built for exactly this lookup). The
two gender concept_ids given as defaults (8507=MALE, 8532=FEMALE) are
OMOP's stable core vocabulary, consistent across virtually every real
OMOP instance — still worth confirming against your own CONCEPT table,
not treated as fundamentally different from the ones you must supply.

A REAL, HONEST GAP THIS ADAPTER SURFACES, NOT HIDES: heart rate
variability (HRV) is rarely captured as a discrete, structured OMOP
Measurement at most institutions — it typically requires raw ECG
waveform analysis, unlike heart rate and even QTc, which are often
directly documented as vitals. Don't assume an hrv_concept_id exists in
your instance. This adapter handles a missing HRV concept_id by leaving
hrv_sdnn_ms out of the feature dict per record rather than failing —
LabelValidator's feature-range check already tolerates a missing
feature (see label_schema.py's validate_record).

A REAL SIMPLIFICATION, ALSO STATED HONESTLY: index observations are
built by grouping Measurement rows sharing the same (person_id, date) —
a common, reasonable heuristic when there's no cleaner link, but a real
deployment should prefer grouping by visit_occurrence_id where that's
populated, since two readings on the same calendar date aren't
necessarily the same clinical encounter. See `group_by_visit` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type, timedelta
from typing import Any, Iterable, Optional

from training.label_schema import MACELabelRecord, MACEEventType, SUBGROUP_DIMENSIONS


@dataclass
class OMOPConceptMapping:
    """Every concept_id this adapter needs — see module docstring for why none of these (besides gender) are supplied as defaults."""

    heart_rate_concept_id: int
    qtc_concept_id: int
    hrv_concept_id: Optional[int] = None  # commonly absent structurally — see module docstring

    mi_condition_concept_ids: set[int] = field(default_factory=set)
    stroke_condition_concept_ids: set[int] = field(default_factory=set)
    cv_death_cause_concept_ids: set[int] = field(default_factory=set)  # matched against the Death table's cause_concept_id

    male_gender_concept_id: int = 8507
    female_gender_concept_id: int = 8532
    race_concept_id_map: dict[int, str] = field(default_factory=dict)      # OMOP race_concept_id -> SUBGROUP_DIMENSIONS race_ethnicity string
    ethnicity_concept_id_map: dict[int, str] = field(default_factory=dict)  # optional refinement (e.g. Hispanic ethnicity overriding the race bucket)

    def validate(self) -> list[str]:
        """Catches an incompletely-filled-in mapping before it silently produces wrong labels — call this before build_mace_records."""
        issues = []
        if not self.mi_condition_concept_ids and not self.stroke_condition_concept_ids and not self.cv_death_cause_concept_ids:
            issues.append("No MACE-qualifying concept_ids supplied at all (MI, stroke, or CV death) — every record would be labeled as having no event, which is almost certainly wrong.")
        if self.hrv_concept_id is None:
            issues.append("hrv_concept_id is None — expected at most institutions (see module docstring); hrv_sdnn_ms will be absent from every record's features.")
        if not self.race_concept_id_map:
            issues.append("race_concept_id_map is empty — every record's race_ethnicity subgroup will be missing, which LabelValidator will flag as an exclusion reason for every record.")
        return issues


def _age_band(year_of_birth: int, index_year: int) -> str:
    age = index_year - year_of_birth
    if age < 50:
        return "<50"
    if age < 65:
        return "50-64"
    return "65+"


def _to_date(value: Any) -> Optional[date_type]:
    """Accepts a date, datetime, or ISO string (Spark/pandas can hand back any of these depending on the query) and normalizes to a plain date."""
    if value is None:
        return None
    if isinstance(value, date_type):
        return value
    if hasattr(value, "date"):  # datetime
        return value.date()
    return date_type.fromisoformat(str(value)[:10])


def group_by_visit(measurements: Iterable[dict[str, Any]], prefer_visit_id: bool = True) -> dict[tuple, list[dict[str, Any]]]:
    """
    Groups Measurement rows into index observations. Prefers
    visit_occurrence_id when present and populated (the cleaner link);
    falls back to (person_id, measurement_date) otherwise — see the
    "real simplification" note in the module docstring for what that
    fallback does and doesn't guarantee.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in measurements:
        person_id = row["person_id"]
        visit_id = row.get("visit_occurrence_id") if prefer_visit_id else None
        key = (person_id, "visit", visit_id) if visit_id else (person_id, "date", _to_date(row["measurement_date"]))
        groups.setdefault(key, []).append(row)
    return groups


def _extract_features(rows: list[dict[str, Any]], mapping: OMOPConceptMapping) -> dict[str, float]:
    features: dict[str, float] = {}
    for row in rows:
        cid = row["measurement_concept_id"]
        value = row.get("value_as_number")
        if value is None:
            continue
        if cid == mapping.heart_rate_concept_id:
            features["heart_rate_bpm"] = float(value)
        elif cid == mapping.qtc_concept_id:
            features["qt_interval_ms"] = float(value)
        elif mapping.hrv_concept_id is not None and cid == mapping.hrv_concept_id:
            features["hrv_sdnn_ms"] = float(value)
    return features


def _subgroups(person: dict[str, Any], mapping: OMOPConceptMapping, index_date: date_type) -> dict[str, str]:
    subgroups: dict[str, str] = {}

    gender_cid = person.get("gender_concept_id")
    if gender_cid == mapping.male_gender_concept_id:
        subgroups["gender"] = "Male"
    elif gender_cid == mapping.female_gender_concept_id:
        subgroups["gender"] = "Female"

    ethnicity_cid = person.get("ethnicity_concept_id")
    race_cid = person.get("race_concept_id")
    if ethnicity_cid is not None and ethnicity_cid in mapping.ethnicity_concept_id_map:
        subgroups["race_ethnicity"] = mapping.ethnicity_concept_id_map[ethnicity_cid]
    elif race_cid is not None and race_cid in mapping.race_concept_id_map:
        subgroups["race_ethnicity"] = mapping.race_concept_id_map[race_cid]

    yob = person.get("year_of_birth")
    if yob is not None:
        subgroups["age_band"] = _age_band(int(yob), index_date.year)

    return subgroups


def _find_outcome(
    person_id: Any, index_date: date_type, conditions: list[dict[str, Any]], deaths: list[dict[str, Any]],
    mapping: OMOPConceptMapping, max_followup_days: int,
) -> tuple[Optional[MACEEventType], Optional[int], bool, int, Optional[int]]:
    """Returns (event_type, days_to_event, censored, follow_up_days, days_since_prior_mace)."""
    person_conditions = [c for c in conditions if c["person_id"] == person_id]
    person_deaths = [d for d in deaths if d["person_id"] == person_id]

    # Washout: look BACKWARD for a qualifying event before the index date.
    days_since_prior_mace = None
    for c in person_conditions:
        c_date = _to_date(c["condition_start_date"])
        if c_date is None or c_date >= index_date:
            continue
        cid = c["condition_concept_id"]
        if cid in mapping.mi_condition_concept_ids or cid in mapping.stroke_condition_concept_ids:
            gap = (index_date - c_date).days
            if days_since_prior_mace is None or gap < days_since_prior_mace:
                days_since_prior_mace = gap

    # Forward-looking outcome: earliest qualifying event ON OR AFTER the index date.
    candidates: list[tuple[int, MACEEventType]] = []
    for c in person_conditions:
        c_date = _to_date(c["condition_start_date"])
        if c_date is None or c_date < index_date:
            continue
        cid = c["condition_concept_id"]
        days = (c_date - index_date).days
        if cid in mapping.mi_condition_concept_ids:
            candidates.append((days, MACEEventType.MYOCARDIAL_INFARCTION))
        elif cid in mapping.stroke_condition_concept_ids:
            candidates.append((days, MACEEventType.STROKE))
    for d in person_deaths:
        d_date = _to_date(d.get("death_date"))
        if d_date is None or d_date < index_date:
            continue
        if d.get("cause_concept_id") in mapping.cv_death_cause_concept_ids:
            candidates.append(((d_date - index_date).days, MACEEventType.CARDIOVASCULAR_DEATH))

    if candidates:
        days_to_event, event_type = min(candidates, key=lambda c: c[0])
        if days_to_event <= max_followup_days:
            return event_type, days_to_event, False, days_to_event, days_since_prior_mace

    # No qualifying event within the follow-up window: censored. Follow-up
    # length is capped at max_followup_days here; a real deployment should
    # use the patient's actual last-observed-alive date if that's tracked
    # and shorter than max_followup_days, rather than assuming full coverage.
    return None, None, True, max_followup_days, days_since_prior_mace


def build_mace_records(
    person_rows: list[dict[str, Any]], measurement_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]], death_rows: list[dict[str, Any]],
    mapping: OMOPConceptMapping, max_followup_days: int = 365,
) -> list[MACELabelRecord]:
    """
    The main entry point. Groups measurements into index observations,
    extracts features, computes the forward-looking outcome and backward-
    looking washout check, maps demographics to subgroups, and returns a
    list of MACELabelRecord objects ready for LabelValidator — nothing
    past this point needs to know these came from OMOP CDM.
    """
    mapping_issues = mapping.validate()
    if mapping_issues:
        import warnings
        for issue in mapping_issues:
            warnings.warn(f"OMOPConceptMapping: {issue}")

    persons_by_id = {p["person_id"]: p for p in person_rows}
    groups = group_by_visit(measurement_rows)

    records: list[MACELabelRecord] = []
    for i, (key, rows) in enumerate(groups.items()):
        person_id = key[0]
        person = persons_by_id.get(person_id)
        if person is None:
            continue  # a measurement with no matching Person row — skip rather than guess

        features = _extract_features(rows, mapping)
        if "heart_rate_bpm" not in features and "qt_interval_ms" not in features:
            continue  # nothing usable in this group

        index_date = _to_date(rows[0]["measurement_date"])
        if index_date is None:
            continue

        event_type, days_to_event, censored, follow_up_days, days_since_prior_mace = _find_outcome(
            person_id, index_date, condition_rows, death_rows, mapping, max_followup_days,
        )

        records.append(MACELabelRecord(
            record_id=f"OMOP-{person_id}-{i}", patient_id=str(person_id), modality="ecg",
            index_timestamp=0.0,  # LabelValidator only needs relative day offsets (days_to_event etc.), not this field
            features=features, event_type=event_type, days_to_event=days_to_event,
            follow_up_days=follow_up_days, censored=censored, days_since_prior_mace=days_since_prior_mace,
            subgroups=_subgroups(person, mapping, index_date),
        ))

    return records
