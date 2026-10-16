"""End-to-end checks for the ECG->MACE pipeline on SYNTHETIC data (plumbing only)."""
import json
import numpy as np
import pytest

from ecg_models.checkpoint import approve_checkpoint, save_checkpoint
from ecg_models.compare import paired_auc_delta
from training.train_ecg_mace import synthetic_examples, train_ecg_mace_net


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    ex12 = synthetic_examples(n=240, length=500, n_leads=12)
    m12, r12 = train_ecg_mace_net(ex12, n_epochs=3)
    ex1 = [dict(e, x=e["x"][:1]) for e in ex12]
    m1, r1 = train_ecg_mace_net(ex1, n_epochs=3, wearable_aug=True)
    cut = {"moderate": 0.3, "high": 0.6}
    for m, r, f in ((m12, r12, "ecg_mace_v1.pt"), (m1, r1, "ecg_mace_1lead_v1.pt")):
        save_checkpoint(m, str(d / f), sample_rate_hz=50, length=500, platt=r["platt"], trained_on="synthetic smoke test",
                        metrics=r["test"], tier_cutoffs=cut)
    return d, r12


def test_synthetic_checkpoint_cannot_be_approved(trained):
    d, _ = trained
    with pytest.raises(ValueError, match="synthetic"):
        approve_checkpoint(str(d / "ecg_mace_v1.pt"), "x", "x")


def test_unapproved_checkpoint_is_not_loaded(trained, monkeypatch):
    d, _ = trained
    from inference.ecg_mace_runtime import build_runtime
    assert build_runtime(str(d)) is None


def _force_approve(d):
    for f in ("ecg_mace_v1.json", "ecg_mace_1lead_v1.json"):
        m = json.loads((d / f).read_text()); m.update(approved_for_live=True, approved_by="test"); (d / f).write_text(json.dumps(m))


def test_runtime_routes_by_lead_count_and_explains(trained):
    d, _ = trained
    _force_approve(d)
    from inference.ecg_mace_runtime import build_runtime
    run = build_runtime(str(d))
    rng = np.random.RandomState(0)
    out12 = run({"waveform": rng.randn(12, 500).tolist(), "sample_rate_hz": 50})
    out1 = run({"waveform": rng.randn(1, 700).tolist(), "sample_rate_hz": 50})     # cropped to model length
    assert out12["inputs_used"] == ["raw_ecg_12_lead"] and out1["inputs_used"] == ["raw_ecg_1_lead"]
    assert 0 <= out12["score"] <= 1 and out12["risk_tier"] in ("low", "moderate", "high")
    assert len(out12["explanation"]["lead_importance"]) == 12 and out12["explanation"]["top_windows"]
    assert run({"waveform": rng.randn(3, 500).tolist()}) is None                      # no model for 3 leads
    assert run({"heart_rate_bpm": 80}) is None                                       # no waveform -> placeholder path


def test_registry_falls_back_to_placeholder_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.ecg_mace_runtime.WEIGHTS_DIR", str(tmp_path))
    from inference.models import ModelRegistry
    out = ModelRegistry().predict("ecg", {"heart_rate_bpm": 120, "qt_interval_ms": 480, "hrv_sdnn_ms": 20})
    assert out["model_version"] == "0.0.0-untrained-placeholder"


def test_paired_comparison_gate():
    rng = np.random.RandomState(1); y = (rng.rand(400) < .3).astype(int)
    good = y + rng.randn(400) * .6; weak = rng.randn(400)
    assert paired_auc_delta(y, good, weak)["beats_baseline"] is True
    assert paired_auc_delta(y, weak, good)["beats_baseline"] is False
