"""BigQuery helpers for the Mayo Workspace (no network, no library needed)."""
import sys
import types
from datetime import date

import pytest

from training.mayo.bq import query_rows, table_ref
from training.mayo.radiology_link import cardiac_mri_study_sql, echo_dicom_study_sql


def test_table_ref_quotes_and_refuses_odd_names():
    assert table_ref("proj-1.ds_a", "FACT_ECG") == "`proj-1.ds_a.FACT_ECG`"
    assert table_ref("", "FACT_ECG") == "`FACT_ECG`"
    for bad in ("proj.ds; DROP TABLE x", "a b", "a..b", "`x`", "x.y.z.w"):
        with pytest.raises(ValueError):
            table_ref(bad, "T")
    with pytest.raises(ValueError):
        table_ref("proj.ds", "T; --")


def test_sql_is_bigquery_standard_sql():
    for sql in (cardiac_mri_study_sql("proj.ds"), echo_dicom_study_sql("proj.ds")):
        assert "`proj.ds.FACT_RADIOLOGY_DICOM_INVENTORY`" in sql and "spark" not in sql.lower()
        assert "proj.ds.FACT_RADIOLOGY_DICOM_INVENTORY " not in sql        # never an unquoted dotted path
    assert "MODALITY = 'US'" in echo_dicom_study_sql("proj.ds")
    with pytest.raises(ValueError):
        cardiac_mri_study_sql("proj.ds; DROP")


def test_query_rows_returns_plain_dicts_and_passes_parameters(monkeypatch):
    seen = {}

    class Row:
        def __init__(self, d): self.d = d
        def items(self): return self.d.items()

    class Job:
        def result(self): return [Row({"PATIENT_CLINIC_NUMBER": "7", "d": date(2026, 1, 2)})]

    class Client:
        def query(self, sql, job_config=None):
            seen["sql"], seen["cfg"] = sql, job_config
            return Job()

    fake = types.ModuleType("google.cloud.bigquery")
    fake.ScalarQueryParameter = lambda n, t, v: (n, t, v)
    fake.QueryJobConfig = lambda query_parameters: types.SimpleNamespace(query_parameters=query_parameters)
    cloud = types.ModuleType("google.cloud"); cloud.bigquery = fake
    google = types.ModuleType("google"); google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake)
    rows = query_rows("SELECT 1", client=Client(), params=[("since", "DATE", date(2026, 1, 1))])
    assert rows == [{"PATIENT_CLINIC_NUMBER": "7", "d": date(2026, 1, 2)}]
    assert seen["cfg"].query_parameters == [("since", "DATE", date(2026, 1, 1))]
