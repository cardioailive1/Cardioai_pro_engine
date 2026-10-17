import json
from datetime import date, timedelta

import numpy as np
import pytest

from claims_risk.checkpoint import (approve_claims_checkpoint, save_claims_checkpoint, set_claims_tier_cutoffs)
from claims_risk.features import FEATURE_NAMES, vectorize
from training.omop_adapter import OMOPConceptMapping
from training.omop_claims_adapter import build_claims_risk_records
from training.train_claims_risk import synthetic_records, train_claims_risk

MAP = OMOPConceptMapping(heart_rate_concept_id=1, qtc_concept_id=2, mi_condition_concept_ids={10}, stroke_condition_concept_ids={11},
                         cv_death_cause_concept_ids={90}, race_concept_id_map={5: "White"})
AS_OF = date(2024, 1, 1)


def d(off):
    return (AS_OF + timedelta(days=off)).isoformat()


def test_vectorize_groups_and_unknowns():
    v = dict(zip(FEATURE_NAMES, vectorize(70, None, ["I10", "E11.9", "i50.9"], None, None)))
    assert v["hypertension"] == 1 and v["diabetes"] == 1 and v["heart_failure"] == 1 and v["ckd"] == 0
    assert v["sex_known"] == 0 and v["visits_known"] == 0 and v["n_distinct_dx"] == 3


def test_omop_adapter_windows_labels_and_leakage():
    persons = [{"person_id": i, "gender_concept_id": 8507, "race_concept_id": 5, "year_of_birth": 1950} for i in range(1, 6)]
    obs = [{"person_id": i, "observation_period_start_date": d(-800), "observation_period_end_date": d(800)} for i in range(1, 5)]
    obs.append({"person_id": 5, "observation_period_start_date": d(-100), "observation_period_end_date": d(800)})   # not covered for lookback
    conds = [
        {"person_id": 1, "condition_concept_id": 3, "condition_source_value": "I50.9", "condition_start_date": d(-30)},   # feature (HF)
        {"person_id": 1, "condition_concept_id": 10, "condition_source_value": "I21.4", "condition_start_date": d(100)},  # outcome MI
        {"person_id": 2, "condition_concept_id": 10, "condition_source_value": "I21.4", "condition_start_date": d(-60)},  # prior MI: feature, not label
        {"person_id": 3, "condition_concept_id": 10, "condition_source_value": "I21.4", "condition_start_date": d(500)},  # beyond horizon
    ]
    visits = [{"person_id": 1, "visit_concept_id": 9201, "visit_start_date": d(-10)}, {"person_id": 1, "visit_concept_id": 9203, "visit_start_date": d(-5)}]
    deaths = [{"person_id": 4, "death_date": d(-1), "cause_concept_id": 90}]      # died before as_of: skipped
    recs = {r.patient_id: r for r in build_claims_risk_records(persons, conds, visits, obs, deaths, MAP, [AS_OF])}
    assert set(recs) == {"1", "2", "3"}                      # 4 died, 5 not covered
    f = lambda r: dict(zip(FEATURE_NAMES, r.x))
    assert recs["1"].label == 1 and f(recs["1"])["heart_failure"] == 1 and f(recs["1"])["inpatient_visits"] == 1 and f(recs["1"])["ed_visits"] == 1
    assert "prior_mi" in FEATURE_NAMES and f(recs["1"])["prior_mi"] == 0          # the future MI is NOT a feature
    assert recs["2"].label == 0 and f(recs["2"])["prior_mi"] == 1
    assert recs["3"].label == 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    r = train_claims_risk(synthetic_records(n=3000), n_estimators=40)
    p = tmp_path_factory.mktemp("w") / "claims_risk_v1.joblib"
    save_claims_checkpoint(r, str(p), trained_on="synthetic smoke test")
    return p, r


def test_training_outputs(trained):
    _, r = trained
    assert 0.5 < r["metrics"]["auc"] < 1 and "delta" in r["comparison_to_baseline"] and r["cutoff_proposals"]
    assert any(k.startswith("gender=") for k in r["subgroup_auc"])


def test_approval_gates_and_runtime(trained, tmp_path):
    p, _ = trained
    with pytest.raises(ValueError, match="synthetic"):
        approve_claims_checkpoint(str(p), "x", "x")
    from inference.claims_runtime import build_claims_runtime
    assert build_claims_runtime(str(p.parent)) is None                      # unapproved: not loaded
    mp = p.with_suffix(".json"); m = json.loads(mp.read_text())
    m.update(trained_on="Mayo OMOP", comparison_to_baseline={"beats_baseline": True}); mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="cutoffs"):
        approve_claims_checkpoint(str(p), "x", "x")
    set_claims_tier_cutoffs(str(p), 0.05, 0.15, "Dr Test", "from table")
    approve_claims_checkpoint(str(p), "Dr Test", "ok")
    run = build_claims_runtime(str(p.parent))
    out = run({"member_age": 78, "history_diagnosis_codes": ["I50.9", "E11.9", "I25.10"], "sex": "M", "inpatient_visits_12m": 2})
    assert 0 <= out["score"] <= 1 and out["risk_tier"] in ("low", "moderate", "high") and out["window_days"] == 365
    assert run({"member_age": 78}) is None                                   # no history -> placeholder path


def test_registry_and_agent_fall_back_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.claims_runtime.WEIGHTS_DIR", str(tmp_path))
    from inference.models import ModelRegistry
    out = ModelRegistry().predict("claims", {"member_age": 70, "risk_flags_count": 3})
    assert out["model_version"] == "0.0.0-untrained-placeholder"


def test_inference_agent_adds_history():
    import asyncio
    from orchestrator.agents import InferenceAgent
    from reports.population_report import PopulationAggregator
    agg = PopulationAggregator(); agg.record("M1", 70, 1, 0.1, "low", claim_id="c1", total_charge_amount=100, diagnosis_codes=["I10"])
    seen = {}
    class Reg:
        def predict(self, modality, record):
            seen.update(record); return None
    asyncio.run(InferenceAgent(Reg(), agg).handle({"normalized": {"modality": "claims", "record": {"member_id": "M1", "member_age": 70, "diagnosis_codes": ["E11.9"]}}}))
    assert seen["history_diagnosis_codes"] == ["E11.9", "I10"]
