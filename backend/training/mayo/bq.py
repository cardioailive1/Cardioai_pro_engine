"""
CardioAI Pro - run BigQuery SQL in the Mayo Clinic Platform Workspace
=====================================================================
Mayo (Adam Resnick, 2026-10-28): the Workspace uses the BigQuery client to run SQL against the provisioned data. The SQL in
`radiology_link` is BigQuery Standard SQL, with the dataset given as `project.dataset` and every table quoted in backticks.

    from training.mayo.bq import query_rows
    rows = query_rows(cardiac_mri_study_sql("<project>.<dataset>"))      # list of plain dicts, as the rest of the pipeline expects

The client library is imported only when a query runs, so nothing else in the project needs it. Rows come back as plain dicts (dates as
datetime.date), which is the shape `omop_adapter`, `ecg_echo_link`, `label_audit` and `radiology_link` already take. The rows hold
patient-level data: keep them inside the Workspace and never write them to a file that leaves it.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

_ID = re.compile(r"^[A-Za-z0-9_\-]+(\.[A-Za-z0-9_\-]+){0,2}$")


def table_ref(schema: str, table: str) -> str:
    """BigQuery table reference: `project.dataset.TABLE` quoted in backticks. `schema` is 'project.dataset' (or 'dataset'), or
    empty when the notebook's default dataset is set. Anything that is not a plain dotted name is refused."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", table):
        raise ValueError(f"bad table name {table!r}")
    if not schema:
        return f"`{table}`"
    if not _ID.match(schema):
        raise ValueError(f"schema must look like project.dataset (letters, digits, _, -, .); got {schema!r}")
    return f"`{schema}.{table}`"


def query_rows(sql: str, *, client: Any = None, params: Optional[Sequence[tuple]] = None) -> list[dict]:
    """Run `sql` and return the rows as plain dicts. params: (name, type, value) tuples for @name query parameters, for example
    ("since", "DATE", date(2026, 1, 1)); use these instead of pasting values into the SQL. `client` is a bigquery.Client; by default
    one is created from the Workspace's own credentials."""
    from google.cloud import bigquery           # imported here so the rest of the project does not need the library

    client = client or bigquery.Client()
    cfg = None
    if params:
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter(n, t, v) for n, t, v in params])
    job = client.query(sql, job_config=cfg) if cfg is not None else client.query(sql)
    return [dict(r.items()) for r in job.result()]
