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
