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


def test_mayo_driver_end_to_end(tmp_path):
    import base64, csv
    from ecg_models.checkpoint import set_tier_cutoffs
    from training.train_ecg_mace_mayo import main
    rng = np.random.RandomState(3); t = np.linspace(0, 10, 500)
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "waveform_b64", "label_window", "heart_rate_bpm", "qt_interval_ms"])
        for i in range(160):
            y = i % 5 == 0
            sig = np.stack([np.sin(2 * np.pi * 1.2 * t) ** 15 * 900 + rng.randn(500) * 60 + y * 80 * np.sin(2 * np.pi * 7 * t) for _ in range(12)])
            w.writerow([f"p{i // 2}", base64.b64encode(sig.astype("<i2").tobytes()).decode(),
                        "early_30_90" if y else "none_observed", 60 + rng.randn() * 8 + 10 * y, 400 + rng.randn() * 20])
    main(["--rows", str(path), "--sample-rate", "50", "--seconds", "10", "--epochs", "2", "--out", str(tmp_path / "w")])
    for f in ("ecg_mace_v1.pt", "ecg_mace_1lead_v1.pt", "cutoff_proposals_12lead.json", "cutoff_proposals_1lead.json"):
        assert (tmp_path / "w" / f).exists()
    props = json.loads((tmp_path / "w" / "cutoff_proposals_12lead.json").read_text())
    assert props and {"cutoff", "test_ppv", "share_flagged"} <= set(props[0])
    meta = set_tier_cutoffs(str(tmp_path / "w" / "ecg_mace_v1.pt"), 0.2, 0.5, "Dr Test", "from table")
    assert meta["tier_cutoffs"]["set_by"] == "Dr Test" and meta["approved_for_live"] is False
