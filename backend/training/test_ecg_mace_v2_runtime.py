"""ECG v2 live-runtime gate: loaded only with approval + matching checksums; otherwise v1 keeps scoring. SYNTHETIC stacker (plumbing only)."""
import hashlib
import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ecg_models.checkpoint import save_checkpoint
from ecg_models.ecg_v2_stacker import HEAD, TAIL, calibrated_logit, load_v2
from training.train_ecg_mace import synthetic_examples, train_ecg_mace_net

CONDS = ["heart_failure", "diabetes", "ckd"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture()
def wdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("w2")
    m, r = train_ecg_mace_net(synthetic_examples(n=240, length=500, n_leads=12), n_epochs=2)
    save_checkpoint(m, str(d / "ecg_mace_v1.pt"), sample_rate_hz=50, length=500, platt=r["platt"], trained_on="synthetic smoke test", metrics=r["test"], tier_cutoffs={"moderate": 0.3, "high": 0.6})
    j = json.loads((d / "ecg_mace_v1.json").read_text()); j.update(approved_for_live=True, approved_by="test"); (d / "ecg_mace_v1.json").write_text(json.dumps(j))
    v = d / "ecg_mace_v2"; v.mkdir()
    feats = HEAD + CONDS + TAIL; rng = np.random.RandomState(0); X = rng.randn(300, len(feats)); y = (X[:, 0] + rng.randn(300) > 0).astype(int)
    sc = StandardScaler().fit(X); gb = HistGradientBoostingClassifier(max_iter=20).fit(sc.transform(X), y)
    joblib.dump({"scaler": sc, "model": gb, "features": feats, "qrs_fill": 95.0}, v / "stacker_v2.joblib")
    card = {"version": "ecg_mace_v2", "frozen_on": "2026-11-16", "approved": False, "network": {"sha256": sha(d / "ecg_mace_v1.pt")},
            "stacker": {"sha256": sha(v / "stacker_v2.joblib"), "features": feats, "qrs_fill_median": 95.0}, "history_features": {"conditions": {c: 1 for c in CONDS}}}
    (v / "frozen_ecg_v2.json").write_text(json.dumps(card))
    (v / "approval_v2.json").write_text(json.dumps({"approved": True, "approved_by": "Dr. Tamanna Nahar; Sampson Kontomah", "stacker_sha256": card["stacker"]["sha256"],
                                                   "network_sha256": card["network"]["sha256"], "tier_cutoffs": {"moderate": 0.2, "high": 0.5}}))
    return d


def _rec(rng, **kw):
    r = {"waveform": rng.randn(12, 500).tolist(), "sample_rate_hz": 50, "age": 70, "sex_male": 1, "heart_rate_bpm": 80, "qt_interval_ms": 420, "qrs_ms": 100,
         "n_acute_visits_365": 1, "n_visits_365": 6, "history": {c: 0 for c in CONDS}}
    r.update(kw); return r


def _run(d):
    from inference.ecg_mace_runtime import build_runtime
    return build_runtime(str(d))


def test_v2_used_when_approved_and_checksums_match(wdir):
    out = _run(wdir)(_rec(np.random.RandomState(1)))
    assert out["model_version"] == "ecg-mace-v2-2026-11-16" and 0 <= out["score"] <= 1 and out["risk_tier"] in ("low", "moderate", "high")
    assert out["calibrated_probability"] is None and "not a stated probability" in out["probability_note"]


def test_v2_missing_inputs_fall_back_to_v1_and_say_why(wdir):
    rec = _rec(np.random.RandomState(1)); rec["history"].pop("ckd"); rec["age"] = None
    out = _run(wdir)(rec)
    assert out["model_version"].startswith("ecg-mace-12lead") and "age" in out["v2_not_used"] and "history.ckd" in out["v2_not_used"]


def test_missing_qrs_uses_frozen_fill(wdir):
    assert _run(wdir)(_rec(np.random.RandomState(1), qrs_ms=None))["model_version"].startswith("ecg-mace-v2")


@pytest.mark.parametrize("breakage", ["no_approval", "approved_false", "stacker_changed", "network_changed", "bad_cutoffs", "no_name"])
def test_v2_refused_unless_every_gate_passes(wdir, breakage):
    v = wdir / "ecg_mace_v2"; a = json.loads((v / "approval_v2.json").read_text())
    if breakage == "no_approval": (v / "approval_v2.json").unlink()
    elif breakage == "approved_false": a["approved"] = False
    elif breakage == "no_name": a["approved_by"] = ""
    elif breakage == "bad_cutoffs": a["tier_cutoffs"] = {"moderate": 0.6, "high": 0.2}
    elif breakage == "stacker_changed": (v / "stacker_v2.joblib").write_bytes((v / "stacker_v2.joblib").read_bytes() + b"x")
    elif breakage == "network_changed": (wdir / "ecg_mace_v1.pt").write_bytes((wdir / "ecg_mace_v1.pt").read_bytes() + b"x")
    if breakage != "no_approval": (v / "approval_v2.json").write_text(json.dumps(a))
    assert load_v2(str(wdir), str(wdir / "ecg_mace_v1.pt")) is None


def test_recalibration_is_monotone_and_only_with_matching_stacker(wdir):
    v = wdir / "ecg_mace_v2"; card = json.loads((v / "frozen_ecg_v2.json").read_text())
    (v / "recalibration_v2.json").write_text(json.dumps({"stacker_sha256": card["stacker"]["sha256"], "slope": 0.9, "intercept_at_natural_prevalence": -4.0}))
    rng = np.random.RandomState(2); run = _run(wdir); outs = [run(_rec(rng, age=a)) for a in (30, 90)]
    assert all(o["calibrated_probability"] is not None and 0 < o["calibrated_probability"] < 0.5 for o in outs)
    (v / "recalibration_v2.json").write_text(json.dumps({"stacker_sha256": "other", "slope": 0.9, "intercept_at_natural_prevalence": -4.0}))
    assert _run(wdir)(_rec(rng))["calibrated_probability"] is None


def test_calibrated_logit_matches_training_formula():
    a, b, l = 0.8, -0.3, 1.7
    p = 1 / (1 + np.exp(-(a * l + b))); assert abs(calibrated_logit(l, {"a": a, "b": b}) - np.log(p / (1 - p))) < 1e-9
