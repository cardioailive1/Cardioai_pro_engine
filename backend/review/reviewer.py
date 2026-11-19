"""
Rules-based reviewer that runs AFTER a prediction, to help a clinician or analyst read the score.

It never changes the score or the tier, never states a diagnosis and never says what treatment to give. It only:
  1. checks the inputs and the result (missing or out-of-range values, placeholder instead of trained model, borderline tier, ...);
  2. says in plain numbers what the tier means (from the locked-set results, so most flags are false alarms);
  3. lists the known limits of that model;
  4. lists questions for the reader to consider;
  5. returns an audit record the caller can store (no patient identifiers are put in it).

Everything is fixed text and fixed checks (no LLM), so the output is predictable and easy to approve. The wording is a DRAFT until
the clinical lead approves it: set REVIEW_WORDING_APPROVED_BY below (by name) and nothing else changes. Not for use as a diagnosis.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from claims_risk.features import DX_GROUPS, normalize_code

REVIEW_VERSION = "review-rules-1"
REVIEW_WORDING_APPROVED_BY: Optional[str] = None          # stays None until the clinical lead approves the wording by name

DISCLAIMER = ("This review explains the score; it is not a diagnosis and does not decide care. A clinician or analyst makes the decision "
              "and may disagree with the score.")

# Numbers below are copied from the locked-set reports (Mayo data). Update them only from a new locked-set result.
ECG_V2 = {
    "moderate": {"found": "about 9 in 10", "ppv_pct": 1.6, "extra_flags": 61},
    "high": {"found": "about 8 in 10", "ppv_pct": 2.3, "extra_flags": 42},
    "limits": ["Tested on Mayo ECGs only (locked 2024-01-01 to 2025-12-27 set); not tested at other sites.",
               "Weakest for non-fatal heart attack or stroke (AUC about 0.72); stronger for death (AUC about 0.87).",
               "Predicts the 30 to 89 days after the ECG; it says nothing about the first 30 days or later than 89 days.",
               "Events are rare (about 1%), so most flagged patients will not have an event."],
}
CLAIMS_V1 = {
    "moderate": {"found": "about 9 in 10", "ppv_pct": "2.5 to 3.3", "flagged": "about half of members"},
    "high": {"found": "about 8 in 10", "ppv_pct": "3.1 to 4.1", "flagged": "about 4 in 10 members"},
    "limits": ["Tested on Mayo health-system data, not payer claims; validate on payer claims before relying on it.",
               "Predicts 12-month first heart attack, ischemic stroke or death (probable cancer deaths excluded).",
               "Ranks less well within an age band (AUC 0.66 to 0.74) than across all adults.",
               "Stated probability drifted about 10% high on 2025 data (still inside the limits); tiers use the score, not the probability.",
               "Not approved for pricing or coverage decisions."],
}
AGE_BAND_AUC = [(18, 44, 0.74), (45, 64, 0.72), (65, 79, 0.68), (80, 200, 0.66)]    # 2025 forward-test AUC by age band


def _chk(code: str, level: str, text: str) -> dict:
    return {"code": code, "level": level, "text": text}                              # level: "info" | "check" | "warning"


def _tier_text(result: dict, table: dict) -> Optional[str]:
    tier = result.get("risk_tier")
    t = table.get(tier)
    if tier == "low":
        return "Low tier: below the moderate cutoff. A low tier lowers the chance of an event; it does not rule one out."
    if not t:
        return None
    base = (f"{tier.capitalize()} tier: on test data this cutoff found {t['found']} of the people who went on to have an event. "
            f"Only about {t['ppv_pct']} in 100 flagged people had one, so most flags are false alarms.")
    if "extra_flags" in t:
        base += f" (about {t['extra_flags']} flagged people without an event for each person with one)"
    return base


def _borderline(score: Any, result: dict, cuts: Optional[dict]) -> Optional[dict]:
    if not cuts or not isinstance(score, (int, float)):
        return None
    for name in ("moderate", "high"):
        c = cuts.get(name)
        if c and abs(score - c) <= 0.1 * c:
            return _chk("borderline_score", "check", f"The score is within 10% of the {name} cutoff, so a small change in an input could move the tier.")
    return None


def _placeholder(result: dict) -> bool:
    return "placeholder" in str(result.get("note", "")).lower()


def _review_claims(record: dict, result: dict) -> tuple[list, list, Optional[str], list]:
    checks: list = []; limits = list(CLAIMS_V1["limits"])
    age = record.get("member_age"); codes = record.get("history_diagnosis_codes")
    if _placeholder(result):
        checks.append(_chk("placeholder_model", "warning", "This score comes from a hand-set placeholder, not the approved trained model. Do not rely on it."))
        return checks, limits, None, ["Is a trained, approved model available for this record (it needs member age and the diagnosis history)?"]
    if age is not None and not (18 <= float(age) <= 110):
        checks.append(_chk("age_outside_tested_range", "warning", "Age is outside 18 to 110; the model was built for adults."))
    if codes is not None and len(codes) == 0:
        checks.append(_chk("no_history_codes", "check", "The diagnosis history is empty. An empty history can mean no care or a missing history, and the score may be low for that reason."))
    if str(record.get("sex", "")).upper() not in ("M", "MALE", "F", "FEMALE"):
        checks.append(_chk("sex_unknown", "check", "Sex is missing; the model used a neutral value."))
    if record.get("inpatient_visits_12m") is None or record.get("ed_visits_12m") is None:
        checks.append(_chk("visit_counts_missing", "check", "Hospital or ED visit counts are missing; the model scored without them."))
    for lo, hi, auc in AGE_BAND_AUC:
        if age is not None and lo <= float(age) <= hi and auc < 0.70:
            checks.append(_chk("weaker_ranking_age_band", "info", f"For ages {lo}-{hi if hi < 200 else '+'} the model ranks less well (AUC about {auc}); read the tier with more caution."))
    present = sorted(g for g, prefixes in DX_GROUPS.items() if codes and any(normalize_code(c).startswith(p) for c in codes for p in prefixes))
    inputs = ([f"Diagnosis groups present on the record (these are model inputs, not a ranked explanation): {', '.join(present)}."] if present
              else ["No tracked diagnosis group is present on the record."])
    if result.get("calibrated_probability") is not None:
        checks.append(_chk("stated_probability", "info", "The stated probability is adjusted to Mayo's all-adult rate and is not a rate for your population."))
    consider = ["Does the member's history here match what you know of them (missing claims lower the score)?",
                "Is this member already in a care program, or have recent events happened outside this record?",
                "Would this member be better assessed by a clinician than by the score alone?"]
    return checks, limits, _tier_text(result, CLAIMS_V1), inputs + ["Questions to consider: " + " ".join(consider)]


def _review_ecg(record: dict, result: dict) -> tuple[list, list, Optional[str], list]:
    checks: list = []; limits = list(ECG_V2["limits"])
    if _placeholder(result):
        checks.append(_chk("placeholder_model", "warning", "This score comes from a hand-set placeholder, not the approved trained model. Do not rely on it."))
        return checks, limits, None, ["Is a trained, approved ECG model available for this record (it needs the raw waveform)?"]
    version = str(result.get("model_version", ""))
    if "v2" not in version:
        checks.append(_chk("v2_not_used", "check", result.get("v2_not_used") or "ECG v2 was not applied; the ECG network alone scored this record, which has not been tested as extensively."))
    hr, qt, qrs, age = (record.get(k) for k in ("heart_rate_bpm", "qt_interval_ms", "qrs_ms", "age"))
    for name, v, lo, hi in (("heart rate", hr, 30, 200), ("QT interval", qt, 250, 600), ("QRS duration", qrs, 60, 220), ("age", age, 18, 110)):
        if v is not None and not (lo <= float(v) <= hi):
            checks.append(_chk(f"{name.split()[0]}_out_of_range", "warning", f"{name.capitalize()} ({v}) is outside {lo} to {hi}; check the value or the recording."))
    if result.get("second_stage") and "echo" in str(result["second_stage"]).lower() and result.get("echo_used") is False:
        checks.append(_chk("echo_not_used", "info", "An earlier echo was not used (" + str(result.get("echo_note", "")).strip() + ")."))
    if result.get("calibrated_probability") is not None:
        checks.append(_chk("probability_not_verified", "check", "The stated probability has not been verified on newer data. Use the tier; do not quote the probability."))
    inputs = []
    ex = result.get("explanation") or {}
    if ex.get("top_windows"):
        w = ex["top_windows"][0]
        inputs.append(f"The network's score changed most when {w['start_s']} to {w['end_s']} seconds of the tracing was hidden. This shows where the model looked, not why the patient is at risk.")
    consider = ["Does the tracing quality look adequate (noise, lead placement)?",
                "Does the score fit the clinical picture, recent symptoms and earlier tests?",
                "Has there been a heart attack or stroke in the last 90 days? The model was built to leave those patients out."]
    return checks, limits, _tier_text(result, {"moderate": ECG_V2["moderate"], "high": ECG_V2["high"], "low": None}), inputs + ["Questions to consider: " + " ".join(consider)]


_BY_MODALITY = {"claims": _review_claims, "ecg": _review_ecg}


def review_prediction(modality: str, record: dict[str, Any], result: Optional[dict[str, Any]], cutoffs: Optional[dict[str, float]] = None) -> Optional[dict[str, Any]]:
    """Return the review for one prediction, or None if this modality has no reviewer. `record` is the model input, `result` its output.
    `cutoffs` ({"moderate": x, "high": y}) is optional; with it the review also flags a borderline score."""
    fn = _BY_MODALITY.get(modality)
    if fn is None or not result or result.get("score") is None:
        return None
    checks, limits, tier_text, points = fn(record or {}, result)
    bl = _borderline(result.get("score"), result, cutoffs)
    if bl:
        checks.append(bl)
    status = "warning" if any(c["level"] == "warning" for c in checks) else "check" if any(c["level"] == "check" for c in checks) else "ok"
    return {"review_version": REVIEW_VERSION, "modality": modality, "status": status, "score": result.get("score"), "risk_tier": result.get("risk_tier"),
            "model_version": result.get("model_version"), "what_the_tier_means": tier_text, "checks": checks, "known_limits": limits,
            "for_the_reader": points, "disclaimer": DISCLAIMER,
            "wording_status": "approved by " + REVIEW_WORDING_APPROVED_BY if REVIEW_WORDING_APPROVED_BY else "DRAFT: wording not yet approved by the clinical lead",
            "reviewed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "reader_decision": None}


def record_decision(review: dict[str, Any], reader: str, decision: str, reason: str = "") -> dict[str, Any]:
    """Audit entry for what the human did with the score. decision: 'accepted' | 'overridden' | 'needs_more_information'.
    Contains no patient identifiers; the caller stores it against its own record id."""
    if decision not in ("accepted", "overridden", "needs_more_information"):
        raise ValueError("decision must be accepted, overridden or needs_more_information")
    if not reader or not reader.strip():
        raise ValueError("the reader must be named")
    if decision == "overridden" and not reason.strip():
        raise ValueError("an override needs a reason")
    return {"review_version": review["review_version"], "model_version": review["model_version"], "score": review["score"], "risk_tier": review["risk_tier"],
            "check_codes": [c["code"] for c in review["checks"]], "reader": reader.strip(), "decision": decision, "reason": reason.strip(),
            "decided_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
