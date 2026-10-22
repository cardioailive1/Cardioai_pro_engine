from datetime import date

import numpy as np

from training.ecg_echo_stage2 import fit_stage2
from training.mayo.ecg_echo_link import echo_study_lvef, link_ecg_to_echo


def _echo(pid, d, v, study=None):
    return {"patient_id": pid, "echo_date": d, "LVEF_PCT": v, "study": study}


def test_echo_after_the_ecg_is_never_used_and_latest_earlier_echo_wins():
    studies, st = echo_study_lvef([_echo("p", date(2024, 1, 1), 60), _echo("p", date(2024, 6, 1), 35),
                                   _echo("p", date(2024, 9, 1), 20)], lvef_column="LVEF_PCT")
    rows, ls = link_ecg_to_echo([{"patient_id": "p", "ecg_date": date(2024, 7, 1)},
                                 {"patient_id": "p", "ecg_date": date(2023, 12, 1)},       # only echoes AFTER it
                                 {"patient_id": "p", "ecg_date": date(2024, 6, 1)},        # same day is allowed
                                 {"patient_id": "q", "ecg_date": date(2024, 7, 1)}], studies)
    assert [r["echo"]["lvef"] for r in rows] == [35.0, None, 35.0, None]
    assert rows[0]["echo"]["days_since_echo"] == 30 and rows[2]["echo"]["days_since_echo"] == 0
    assert ls["with_echo"] == 2 and ls["echo_only_after_the_ecg_not_used"] == 1 and ls["ecgs"] == 4
    assert not rows[1]["echo"]["has_echo"]                                                # kept, not dropped


def test_window_limit_and_value_cleaning():
    studies, st = echo_study_lvef([_echo("p", date(2023, 1, 1), 50),
                                   _echo("a", date(2024, 1, 1), 55, "s1"), _echo("a", date(2024, 1, 1), 45, "s1"),   # two values, one study
                                   _echo("b", date(2024, 1, 1), 3), _echo("b", date(2024, 1, 2), "n/a"), _echo("b", None, 50),
                                   _echo("c", date(2024, 1, 1), "0.62")], lvef_column="LVEF_PCT", study_column="study")
    by = {s["patient_id"]: s for s in studies}
    assert by["a"]["lvef"] == 50.0 and by["a"]["n_values"] == 2 and by["c"]["lvef"] == 62.0 and "b" not in by
    assert st["dropped_implausible"] == 1 and st["dropped_not_a_number"] == 1 and st["dropped_no_date"] == 1
    _, ls = link_ecg_to_echo([{"patient_id": "p", "ecg_date": date(2024, 6, 1)}], studies, max_days_before=365)
    assert ls["with_echo"] == 0          # 517 days earlier: outside the window


def _cohort(n, lvef_matters, seed):
    rng = np.random.RandomState(seed)
    latent = rng.randn(n)
    low = rng.rand(n) < 0.15
    logit = -2.2 + 1.0 * latent + (1.6 * low if lvef_matters else 0.0)
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    ecg = 1 / (1 + np.exp(-(-2.0 + 0.9 * latent + 0.6 * rng.randn(n))))            # the network sees only the ECG signal
    has = rng.rand(n) < 0.6
    lv = np.where(low, rng.uniform(20, 40, n), rng.uniform(50, 70, n))
    ex = [{"y": int(y[i]), "context": np.array([1.0, 0.0]),
           "echo": {"has_echo": bool(has[i]), "lvef": float(lv[i]) if has[i] else None, "days_since_echo": 30 if has[i] else None}} for i in range(n)]
    return ex, ecg


def test_lvef_helps_when_it_matters_and_does_not_when_it_does_not():
    for matters in (True, False):
        v, sv = _cohort(4000, matters, 1)
        t, st = _cohort(4000, matters, 2)
        r = fit_stage2(v, sv, t, st)
        assert r["ok"] and 0.5 < r["share_with_echo"] < 0.7
        if matters:
            assert r["beats_ecg_only"] and r["vs_ecg_only_patients_with_echo"]["delta"] > r["vs_ecg_only_all_test_patients"]["delta"] > 0
            assert r["coefficients"]["lvef_le_40"] > 0.5
        else:
            assert not r["beats_ecg_only"]


def test_refuses_with_too_few_events():
    v, sv = _cohort(40, True, 3)
    r = fit_stage2(v, sv, v, sv)
    assert r["ok"] is False and "events" in r["reason"]


def test_column_candidates_several_columns_and_several_values():
    import pytest
    from training.mayo.ecg_echo_link import find_lvef_columns
    assert find_lvef_columns(["PATIENT_DK", "LVEF_BIPLANE", "EJECTION_FRACTION", "EF_VISUAL", "REFERENCE", "ECHO_DTM"]) == ["LVEF_BIPLANE", "EJECTION_FRACTION", "EF_VISUAL"]
    rows = [{"patient_id": "a", "echo_date": date(2024, 1, 1), "BIPLANE": None, "VISUAL": 40},
            {"patient_id": "a", "echo_date": date(2024, 1, 1), "BIPLANE": 50, "VISUAL": 45},
            {"patient_id": "a", "echo_date": date(2024, 1, 1), "BIPLANE": 60, "VISUAL": None}]
    med, _ = echo_study_lvef(rows, lvef_column=["BIPLANE", "VISUAL"])
    avg, _ = echo_study_lvef(rows, lvef_column=["BIPLANE", "VISUAL"], multi_value="mean")
    assert med[0]["lvef"] == 50.0 and med[0]["n_values"] == 3 and abs(avg[0]["lvef"] - 50.0) < 1e-9
    with pytest.raises(ValueError):
        echo_study_lvef(rows, lvef_column="BIPLANE", multi_value="max")
