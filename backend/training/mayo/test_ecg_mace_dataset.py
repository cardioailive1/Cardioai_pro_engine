import base64
import numpy as np
from training.mayo.ecg_mace_dataset import build_examples


def _b64(arr):
    return base64.b64encode(arr.astype("<i2").tobytes()).decode()


def test_build_examples_filters_windows_and_shapes():
    sig = (np.random.RandomState(0).randn(12, 800) * 100)
    rows = [
        {"patient_id": "a", "waveform_b64": _b64(sig), "label_window": "early_30_90", "age": 70, "sex_male": 1},
        {"patient_id": "b", "waveform_b64": _b64(sig), "label_window": "none_observed"},
        {"patient_id": "c", "waveform_b64": _b64(sig), "label_window": "imminent_0_30"},
        {"patient_id": "d", "waveform_b64": "AAAA", "label_window": "none_observed"},
    ]
    ex, skipped = build_examples(rows, length=1000, dtype="int16", n_leads=12, layout="lead_major")
    assert [e["y"] for e in ex] == [1, 0]
    assert ex[0]["x"].shape == (12, 1000)
    assert skipped == {"excluded_window": 1, "bad_waveform": 1}


def test_single_lead_selection():
    sig = np.random.RandomState(0).randn(12, 800) * 100
    rows = [{"patient_id": "a", "waveform_b64": _b64(sig), "label_window": "none_observed"}]
    ex, _ = build_examples(rows, length=500, leads=[0], dtype="int16", n_leads=12, layout="lead_major")
    assert ex[0]["x"].shape == (1, 500)


def test_eight_stored_leads_are_expanded_to_twelve():
    import base64
    import numpy as np
    from training.label_schema import LabelWindow
    from training.mayo.ecg_mace_dataset import build_examples
    raw = np.random.RandomState(0).randint(-500, 500, size=(8, 300)).astype("<i2")
    row = {"patient_id": "p", "waveform_b64": base64.b64encode(raw.tobytes()).decode(), "label_window": LabelWindow.NONE_OBSERVED.value}
    ex, _ = build_examples([row], length=300, dtype="int16", n_leads=8, layout="lead_major")
    assert ex[0]["x"].shape == (12, 300)
    one, _ = build_examples([row], length=300, leads=[0], dtype="int16", n_leads=8, layout="lead_major")
    assert one[0]["x"].shape == (1, 300)


# ---- Mayo FACT_ECG_WAVEFORM (confirmed by Mayo 2026-10-21)
def _mayo_value(counts):
    from urllib.parse import quote
    return quote(base64.b64encode(np.asarray(counts, dtype="<i2").tobytes()).decode(), safe="")


def _mayo_records(n=500, extra=None, wtype="Rhythm"):
    rng = np.random.RandomState(1)
    recs = [{"LEAD_ID": l, "WAVEFORM_TYPE": wtype, "WAVEFORM": _mayo_value(rng.randint(-300, 300, n))}
            for l in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
    return recs + (extra or [])


def test_decode_mayo_lead_url_decodes_and_scales_to_microvolts():
    from training.mayo.ecg_waveform import decode_mayo_lead
    counts = np.array([-2, -1, 0, 1, 1000, 32767], dtype="<i2")
    v = _mayo_value(counts)
    assert "%" in v                                  # '=' padding is URL-encoded, as Mayo's values are
    np.testing.assert_allclose(decode_mayo_lead(v), counts.astype(np.float32) * 4.88, rtol=1e-6)


def test_assemble_keeps_rhythm_only_and_never_joins_records():
    import pytest
    from training.mayo.ecg_waveform import assemble_mayo_ecg
    median = {"LEAD_ID": "I", "WAVEFORM_TYPE": "Median", "WAVEFORM": _mayo_value(np.ones(50))}
    sig = assemble_mayo_ecg(_mayo_records(extra=[median]))
    assert sig.shape == (8, 500)                     # the median beat was ignored
    dup = {"LEAD_ID": "I", "WAVEFORM_TYPE": "Rhythm", "WAVEFORM": _mayo_value(np.ones(500))}
    with pytest.raises(ValueError, match="no chunk order"):
        assemble_mayo_ecg(_mayo_records(extra=[dup]))
    with pytest.raises(ValueError, match="missing leads"):
        assemble_mayo_ecg(_mayo_records()[:7])


def test_resample_250_to_500_doubles_samples_and_keeps_shape():
    from training.mayo.ecg_waveform import resample
    t = np.arange(2500) / 250.0
    sig = np.vstack([np.sin(2 * np.pi * 2 * t)] * 8).astype(np.float32)
    up = resample(sig, 250, 500)
    assert up.shape == (8, 5000)
    assert abs(float(up[0, 1000]) - float(np.sin(2 * np.pi * 2 * (1000 / 500)))) < 0.02
    assert resample(up, 500, 250).shape == (8, 2500)


def test_build_examples_from_mayo_rows_mixed_rates():
    from training.label_schema import LabelWindow
    rows = []
    for pid, hz, n in (("a", 500, 5000), ("b", 250, 2500)):
        recs = _mayo_records(n)
        rows.append({"patient_id": pid, "leads": {r["LEAD_ID"]: r["WAVEFORM"] for r in recs}, "sample_rate_hz": hz,
                     "label_window": LabelWindow.NONE_OBSERVED.value})
    rows.append({"patient_id": "c", "leads": {"I": _mayo_value([1, 2, 3])}, "sample_rate_hz": 500,
                 "label_window": LabelWindow.NONE_OBSERVED.value})
    ex, skipped = build_examples(rows, length=5000, target_rate_hz=500)
    assert [e["x"].shape for e in ex] == [(12, 5000), (12, 5000)]
    assert skipped["bad_waveform"] == 1
