"""discover_schema: metadata only, no network and no BigQuery library needed."""
import csv
import json
from types import SimpleNamespace as NS

import pytest

from scripts.discover_schema import discover, main


class FakeClient:
    def __init__(self):
        self.queried = False

    def list_datasets(self, project):
        return [NS(dataset_id="ds_a"), NS(dataset_id="ds_denied")]

    def list_tables(self, ref):
        if ref.endswith("ds_denied"):
            raise PermissionError("403 access denied")
        return [NS(table_id="FACT_ECG_WAVEFORM", table_type="TABLE")]

    def get_table(self, ref):
        leaf = NS(name="x", field_type="INTEGER", mode="NULLABLE", description="", fields=[])
        rec = NS(name="rec", field_type="RECORD", mode="REPEATED", description="", fields=[leaf])
        lead = NS(name="LEAD_ID", field_type="STRING", mode="NULLABLE", description="lead", fields=[])
        return NS(num_rows=123, schema=[lead, rec])

    def query(self, *a, **k):
        self.queried = True
        raise AssertionError("must never run a query")


def test_discover_is_metadata_only_and_records_errors():
    c = FakeClient()
    inv = discover(c, "mcp-ss-data-p-5o6i")
    assert not c.queried
    a, denied = inv["datasets"]
    assert a["tables"][0]["rows"] == 123
    assert [x["column"] for x in a["tables"][0]["columns"]] == ["LEAD_ID", "rec", "rec.x"]
    assert "PermissionError" in denied["error"] and denied["tables"] == []


def test_refuses_bad_ids():
    for p in ("", "BAD PROJECT", "x"):
        with pytest.raises(ValueError):
            discover(FakeClient(), p)
    with pytest.raises(ValueError):
        discover(FakeClient(), "mcp-ss-data-p-5o6i", "ds; DROP")


def test_main_writes_files(tmp_path, capsys):
    out = tmp_path / "inv"
    assert main(["--project", "mcp-ss-data-p-5o6i", "--out", str(out)], client=FakeClient()) == 0
    assert json.loads((out / "schema_inventory.json").read_text())["project"] == "mcp-ss-data-p-5o6i"
    rows = list(csv.DictReader((out / "columns.csv").open()))
    assert {r["column"] for r in rows} == {"LEAD_ID", "rec", "rec.x"}
    assert "ds_a: 1 tables, 3 columns" in capsys.readouterr().out
