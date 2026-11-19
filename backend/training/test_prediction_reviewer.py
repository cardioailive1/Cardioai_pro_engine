"""Rules-based prediction reviewer: wording limits, input checks, audit entry, and that it never changes a prediction."""
import re

import pytest

from inference.models import ModelRegistry
from review import record_decision, review_prediction

CLAIMS_OUT = {"model": "population_risk_stratifier", "model_version": "claims-risk-test", "score": 0.02, "risk_tier": "high", "calibrated_probability": 0.03,
              "note": "Approved checkpoint trained on Mayo OMOP history."}
CLAIMS_REC = {"member_age": 72, "sex": "F", "history_diagnosis_codes": ["I10", "E11.9", "I50.9"], "inpatient_visits_12m": 1, "ed_visits_12m": 0}
ECG_OUT = {"model": "ecg_mace_risk", "model_version": "ecg-mace-v2-test", "score": 0.05, "risk_tier": "moderate", "calibrated_probability": 0.02,
           "explanation": {"top_windows": [{"start_s": 2.0, "end_s": 2.5, "logit_drop": 0.4}]}, "note": "ECG v2"}
ECG_REC = {"age": 70, "sex_male": True, "heart_rate_bpm": 78, "qt_interval_ms": 420, "qrs_ms": 96}
BANNED = re.compile(r"\b(you have|patient has|diagnos(?:is|ed) (?:of|with)|should (?:take|start|stop|be treated)|prescribe|start treatment|treatment plan)\b", re.I)


def _all_text(rv):
    return " ".join([rv["what_the_tier_means"] or "", rv["disclaimer"], *rv["known_limits"], *rv["for_the_reader"], *(c["text"] for c in rv["checks"])])


def test_claims_review_explains_tier_limits_and_inputs():
    rv = review_prediction("claims", CLAIMS_REC, CLAIMS_OUT)
    assert rv["status"] == "ok" and "false alarms" in rv["what_the_tier_means"]
    assert any("payer claims" in t for t in rv["known_limits"])
    assert any("hypertension" in t and "diabetes" in t and "heart_failure" in t for t in rv["for_the_reader"])
    assert rv["wording_status"].startswith("DRAFT") and rv["reader_decision"] is None


def test_claims_checks_missing_inputs():
    rv = review_prediction("claims", {"member_age": 130, "history_diagnosis_codes": []}, CLAIMS_OUT)
    codes = {c["code"] for c in rv["checks"]}
    assert {"age_outside_tested_range", "no_history_codes", "sex_unknown", "visit_counts_missing"} <= codes and rv["status"] == "warning"


def test_placeholder_is_called_out_and_not_explained_as_model():
    rv = review_prediction("claims", {"member_age": 50}, {**CLAIMS_OUT, "note": "Placeholder heuristic pending trained model — not clinically validated."})
    assert rv["checks"][0]["code"] == "placeholder_model" and rv["what_the_tier_means"] is None


def test_ecg_review_flags_v1_fallback_and_out_of_range():
    out = {**ECG_OUT, "model_version": "ecg-mace-12lead-test", "v2_not_used": "v2 needs these inputs and did not get them: heart_rate_bpm"}
    rv = review_prediction("ecg", {**ECG_REC, "heart_rate_bpm": 260}, out)
    codes = {c["code"] for c in rv["checks"]}
    assert "v2_not_used" in codes and "heart_out_of_range" in codes and rv["status"] == "warning"
    assert any("heart_rate_bpm" in c["text"] for c in rv["checks"] if c["code"] == "v2_not_used")


def test_ecg_review_probability_and_explanation_caveat():
    rv = review_prediction("ecg", ECG_REC, ECG_OUT)
    assert any(c["code"] == "probability_not_verified" for c in rv["checks"])
    assert any("where the model looked" in t for t in rv["for_the_reader"]) and "9 in 10" in rv["what_the_tier_means"]


def test_low_tier_text_never_rules_out_risk():
    rv = review_prediction("claims", CLAIMS_REC, {**CLAIMS_OUT, "risk_tier": "low", "score": 0.002})
    assert "does not rule one out" in rv["what_the_tier_means"]


def test_borderline_with_cutoffs():
    rv = review_prediction("claims", CLAIMS_REC, {**CLAIMS_OUT, "score": 0.0145, "risk_tier": "moderate"}, cutoffs={"moderate": 0.0092, "high": 0.0147})
    assert any(c["code"] == "borderline_score" for c in rv["checks"])


def test_wording_never_diagnoses_or_directs_treatment():
    for mod, rec, out in (("claims", CLAIMS_REC, CLAIMS_OUT), ("ecg", ECG_REC, ECG_OUT), ("claims", {}, CLAIMS_OUT)):
        for tier in ("low", "moderate", "high"):
            assert not BANNED.search(_all_text(review_prediction(mod, rec, {**out, "risk_tier": tier})))
    assert "not a diagnosis" in review_prediction("ecg", ECG_REC, ECG_OUT)["disclaimer"]


def test_no_reviewer_for_other_modalities_or_empty_result():
    assert review_prediction("wearable", {}, {"score": 0.1}) is None and review_prediction("ecg", {}, None) is None and review_prediction("ecg", {}, {"risk_tier": "low"}) is None


def test_record_decision_rules():
    rv = review_prediction("ecg", ECG_REC, ECG_OUT)
    d = record_decision(rv, "Dr Example", "overridden", "tracing is noisy")
    assert d["decision"] == "overridden" and d["reader"] == "Dr Example" and "patient" not in " ".join(d)
    with pytest.raises(ValueError): record_decision(rv, "Dr Example", "overridden", "")
    with pytest.raises(ValueError): record_decision(rv, "  ", "accepted")
    with pytest.raises(ValueError): record_decision(rv, "Dr Example", "approved")


def test_registry_adds_review_without_changing_prediction(monkeypatch):
    reg = ModelRegistry()
    out = reg.predict("claims", {"member_age": 60, "risk_flags_count": 2})
    assert "review" in out and out["review"]["checks"][0]["code"] == "placeholder_model"
    monkeypatch.setenv("CARDIOAI_REVIEW", "0")
    plain = reg.predict("claims", {"member_age": 60, "risk_flags_count": 2})
    assert "review" not in plain and {k: v for k, v in out.items() if k != "review"} == plain     # same score, tier and fields with the review off


def test_review_failure_never_blocks_prediction(monkeypatch):
    import review.reviewer as r
    monkeypatch.setattr(r, "review_prediction", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    import review
    monkeypatch.setattr(review, "review_prediction", r.review_prediction)
    out = ModelRegistry().predict("claims", {"member_age": 60, "risk_flags_count": 2})
    assert out["score"] is not None and "review" not in out
