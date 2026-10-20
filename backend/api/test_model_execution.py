"""The dashboards' model-execution endpoints, with the placeholder and with approved (test-approved) checkpoints.
Synthetic data: this proves the plumbing, not model quality."""
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from orchestrator.orchestrator import orchestrator
from inference.ecg_leads import MAYO_LEADS, derive_twelve_lead, to_lead_array

app = FastAPI(); app.include_router(router, prefix="/api")
client = TestClient(app)


def test_eight_leads_become_twelve_with_standard_identities():
    e = np.random.RandomState(0).randn(8, 50)
    t = derive_twelve_lead(e)
    assert t.shape == (12, 50)
    I, II = e[0], e[1]
    assert np.allclose(t[2], II - I) and np.allclose(t[3], -(I + II) / 2) and np.allclose(t[4], I - II / 2) and np.allclose(t[5], II - I / 2)
    assert np.allclose(t[6:], e[2:])
    # Einthoven's law holds for the derived set, aVR+aVL+aVF = 0
    assert np.allclose(t[3] + t[4] + t[5], 0, atol=1e-5)


def test_lead_input_forms_and_errors():
    d = {k: [0.0] * 300 for k in MAYO_LEADS}
    assert to_lead_array(d).shape == (12, 300)
    assert to_lead_array({"I": [0.0] * 300}).shape == (1, 300)
    for bad in ({"X": [1, 2]}, [[1.0] * 10] * 3, {"I": [1, 2], "II": [1]}):
        with pytest.raises(ValueError):
            to_lead_array(bad)


def test_status_reports_placeholder_when_no_weights(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.model_status.WEIGHTS_DIR", str(tmp_path))
    body = client.get("/api/models/status").json()
    assert body["any_live"] is False and {m["state"] for m in body["models"]} == {"placeholder"}


def test_status_distinguishes_unapproved_and_runtime_missing(tmp_path, monkeypatch):
    from inference import model_status as ms
    (tmp_path / "claims_risk_v1.joblib").write_bytes(b"x")
    side = tmp_path / "claims_risk_v1.json"
    side.write_text(json.dumps({"approved_for_live": False, "trained_on": "x"}))
    assert ms.model_status(str(tmp_path))["models"][3]["state"] == "trained_not_approved"
    side.write_text(json.dumps({"approved_for_live": True, "approved_by": "Dr X", "trained_on": "x", "tier_cutoffs": {"moderate": .1, "high": .2}}))
    assert ms.model_status(str(tmp_path))["models"][3]["state"] == "live"
    monkeypatch.setattr(ms, "_have", lambda pkgs: ["sklearn"])
    assert ms.model_status(str(tmp_path))["models"][3]["state"] == "runtime_missing"


def test_mace_score_without_model_does_not_pretend_to_use_the_waveform():
    wf = {k: np.random.randn(500).tolist() for k in MAYO_LEADS}
    r = client.post("/api/mace/score", json={"waveform": wf, "sample_rate_hz": 250}).json()
    assert r["status"] == "awaiting_trained_model" and r["prediction"] is None and r["n_leads_received"] == 12
    r2 = client.post("/api/mace/score", json={"waveform": wf, "sample_rate_hz": 250, "heart_rate_bpm": 130, "qt_interval_ms": 480}).json()
    assert r2["status"] == "placeholder" and "NOT used" in r2["message"]


def test_mace_score_validates_input():
    assert client.post("/api/mace/score", json={"waveform": [[1.0] * 10] * 3}).status_code == 400
    assert client.post("/api/mace/score", json={"waveform": {"I": [0.0] * 500}, "sample_rate_hz": 5}).status_code == 400
    assert client.post("/api/mace/score", json={"waveform": {"I": [0.0] * 20}}).status_code == 400


def test_mace_score_uses_approved_model_and_logs_to_chart(tmp_path, monkeypatch):
    from ecg_models.checkpoint import save_checkpoint
    from training.train_ecg_mace import synthetic_examples, train_ecg_mace_net
    from inference.ecg_mace_runtime import build_runtime
    ex = synthetic_examples(n=200, length=500, n_leads=12)
    m, r = train_ecg_mace_net(ex, n_epochs=2)
    save_checkpoint(m, str(tmp_path / "ecg_mace_v1.pt"), sample_rate_hz=250, length=500, platt=r["platt"], trained_on="synthetic smoke test",
                    metrics=r["test"], tier_cutoffs={"moderate": .3, "high": .6})
    j = tmp_path / "ecg_mace_v1.json"; meta = json.loads(j.read_text()); meta.update(approved_for_live=True, approved_by="test"); j.write_text(json.dumps(meta))
    monkeypatch.setitem(orchestrator.model_registry._trained, "ecg", build_runtime(str(tmp_path)))
    orchestrator.patient_registry.get_or_create("MRN-T1")
    wf = {k: np.random.randn(1000).tolist() for k in MAYO_LEADS}                 # 500 Hz, 2 s: resampled and cropped to the model length
    r = client.post("/api/mace/score", json={"patient_id": "MRN-T1", "waveform": wf, "sample_rate_hz": 500}).json()
    assert r["status"] == "ok" and r["prediction"]["inputs_used"] == ["raw_ecg_12_lead"] and 0 <= r["prediction"]["score"] <= 1
    assert r["prediction"]["explanation"]["lead_importance"] and r["order"]["status"] == "completed"
    one = client.post("/api/mace/score", json={"waveform": {"I": np.random.randn(500).tolist()}, "sample_rate_hz": 250}).json()
    assert one["status"] == "awaiting_trained_model"                              # no single-lead model approved here


def test_member_scoring_placeholder_then_trained(tmp_path, monkeypatch):
    members = [{"member_id": f"M{i}", "member_age": 60 + i, "sex": "M", "diagnosis_codes": ["I10", "E11.9"] if i % 2 else []} for i in range(6)]
    r = client.post("/api/payer/members/score", json={"members": members}).json()
    assert r["scored"] == 6 and r["placeholder_scored"] == 6 and "placeholder" in r["message"]
    assert orchestrator.population_aggregator.get_member("M3") is not None
    from claims_risk.checkpoint import save_claims_checkpoint
    from training.train_claims_risk import synthetic_records, train_claims_risk
    from inference.claims_runtime import build_claims_runtime
    res = train_claims_risk(synthetic_records(n=2000), n_estimators=30)
    p = tmp_path / "claims_risk_v1.joblib"; save_claims_checkpoint(res, str(p), trained_on="synthetic smoke test")
    j = p.with_suffix(".json"); meta = json.loads(j.read_text()); meta.update(approved_for_live=True, approved_by="test", tier_cutoffs={"moderate": .1, "high": .3}); j.write_text(json.dumps(meta))
    monkeypatch.setitem(orchestrator.model_registry._trained, "claims", build_claims_runtime(str(tmp_path)))
    r2 = client.post("/api/payer/members/score", json={"members": members, "add_to_population_report": False}).json()
    assert r2["placeholder_scored"] == 3                                          # members with no diagnosis history still get the placeholder
    assert "Trained model used" in r2["message"]
    assert client.post("/api/payer/members/score", json={"members": []}).status_code == 400


def test_controlled_cost_avoidance_endpoint():
    rng = np.random.RandomState(4); rows = []
    for i in range(400):
        age = rng.normal(65, 8); pre = max(50, rng.normal(400, 90) + 3 * (age - 65)); treated = bool(rng.rand() < 0.4)
        rows.append({"member_id": f"C{i}", "treated": treated, "covariates": {"age": age}, "pre_pmpm": pre, "post_pmpm": pre + 20 - (60 if treated else 0) + rng.normal(0, 40)})
    r = client.post("/api/payer/cost-avoidance-controlled", json={"members": rows}).json()
    assert "reportable" in r and r["n_treated"] > 100
    assert client.post("/api/payer/cost-avoidance-controlled", json={"members": rows, "design": "x"}).status_code == 400
    bad = rows[:2] + [{**rows[2], "covariates": {"other": 1.0}}]
    assert client.post("/api/payer/cost-avoidance-controlled", json={"members": bad}).status_code == 400
    small = client.post("/api/payer/cost-avoidance-controlled", json={"members": rows[:10]}).json()
    assert small["reportable"] is False and "need at least" in small["reason"]
