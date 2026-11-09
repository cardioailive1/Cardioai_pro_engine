"""
CardioAI Pro — Mayo Clinic Platform schema inventory
====================================================
Lists every dataset, table and column the Workspace's BigQuery client can see in one project, so we can check what Mayo has
provisioned against what the pipeline needs (for example FACT_ECG_WAVEFORM, its LEAD_ID / WAVEFORM_TYPE / WAVEFORM columns).

It reads METADATA ONLY: dataset, table and column names, types, modes and row counts. It never runs a SELECT on table contents, so no
patient-level data is read or written. The output is still kept inside the Workspace: do not copy it out without checking the Mayo
data-use terms.

Run from the backend folder (or, in a notebook, after `os.chdir` to it):

    python -m scripts.discover_schema --project mcp-ss-data-p-5o6i --out exports/schema_inventory
    python -m scripts.discover_schema --project mcp-ss-data-p-5o6i --dataset SOME_DATASET --out exports/schema_inventory

Writes, in --out:  schema_inventory.json, tables.csv, columns.csv   and prints a one-line summary per dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PROJECT = re.compile(r"^[a-z][a-z0-9\-]{4,61}[a-z0-9]$")
_DATASET = re.compile(r"^[A-Za-z0-9_]{1,1024}$")


def _field_rows(fields: Any, prefix: str = "") -> list[dict]:
    """Flatten a table schema (including nested RECORD fields) into one dict per column."""
    out = []
    for f in fields or []:
        name = f"{prefix}{f.name}"
        out.append({"column": name, "type": f.field_type, "mode": f.mode, "description": f.description or ""})
        if getattr(f, "fields", None):
            out.extend(_field_rows(f.fields, prefix=f"{name}."))
    return out


def discover(client: Any, project: str, dataset: Optional[str] = None) -> dict:
    """Return {project, datasets: [{dataset, tables: [{table, type, rows, columns: [...]}]}]}. Metadata only.
    A dataset or table that cannot be read is recorded with an 'error' instead of stopping the run."""
    if not _PROJECT.match(project):
        raise ValueError(f"project id looks wrong: {project!r} (6-63 lowercase letters, digits or dashes, starting with a letter)")
    if dataset is not None and not _DATASET.match(dataset):
        raise ValueError(f"dataset id looks wrong: {dataset!r}")

    if dataset:
        ds_ids = [dataset]
    else:
        ds_ids = [d.dataset_id for d in client.list_datasets(project)]

    result: dict = {"project": project, "datasets": []}
    for ds in ds_ids:
        entry: dict = {"dataset": ds, "tables": []}
        try:
            tables = list(client.list_tables(f"{project}.{ds}"))
        except Exception as e:                                   # permission or not-found: record it and carry on
            entry["error"] = f"{type(e).__name__}: {e}"
            result["datasets"].append(entry)
            continue
        for t in tables:
            row: dict = {"table": t.table_id, "type": getattr(t, "table_type", "")}
            try:
                meta = client.get_table(f"{project}.{ds}.{t.table_id}")
                row["rows"] = getattr(meta, "num_rows", None)
                row["columns"] = _field_rows(meta.schema)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["columns"] = []
            entry["tables"].append(row)
        result["datasets"].append(entry)
    return result


def write_outputs(inv: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / "schema_inventory.json"
    p.write_text(json.dumps(inv, indent=2, default=str), encoding="utf-8")
    paths.append(p)

    p = out_dir / "tables.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "dataset", "table", "type", "rows", "n_columns", "error"])
        for d in inv["datasets"]:
            if d.get("error") and not d["tables"]:
                w.writerow([inv["project"], d["dataset"], "", "", "", "", d["error"]])
            for t in d["tables"]:
                w.writerow([inv["project"], d["dataset"], t["table"], t.get("type", ""), t.get("rows", ""),
                            len(t.get("columns", [])), t.get("error", "")])
    paths.append(p)

    p = out_dir / "columns.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "dataset", "table", "column", "type", "mode", "description"])
        for d in inv["datasets"]:
            for t in d["tables"]:
                for c in t.get("columns", []):
                    w.writerow([inv["project"], d["dataset"], t["table"], c["column"], c["type"], c["mode"], c["description"]])
    paths.append(p)
    return paths


def main(argv: Optional[list[str]] = None, client: Any = None) -> int:
    ap = argparse.ArgumentParser(description="Metadata-only schema inventory of a BigQuery project (no table contents are read).")
    ap.add_argument("--project", required=True, help="BigQuery project id, e.g. mcp-ss-data-p-5o6i")
    ap.add_argument("--dataset", help="limit to one dataset (default: every dataset the client can list)")
    ap.add_argument("--out", default="exports/schema_inventory", help="output folder (default exports/schema_inventory)")
    a = ap.parse_args(argv)

    if client is None:
        from google.cloud import bigquery               # imported here so the rest of the project does not need the library
        client = bigquery.Client(project=a.project)

    inv = discover(client, a.project, a.dataset)
    paths = write_outputs(inv, Path(a.out))

    if not inv["datasets"]:
        print(f"No datasets visible in {a.project}. The data may not be provisioned to this account yet.")
    for d in inv["datasets"]:
        if d.get("error"):
            print(f"{d['dataset']}: cannot list tables ({d['error']})")
        else:
            n_cols = sum(len(t.get("columns", [])) for t in d["tables"])
            print(f"{d['dataset']}: {len(d['tables'])} tables, {n_cols} columns")
    print("wrote:", ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
