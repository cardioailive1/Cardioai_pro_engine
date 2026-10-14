"""
CardioAI Pro - annotator review of the automatic annotations
============================================================
Every automatic label is reviewed by a human before training:
  * all needs_review items go to the queue;
  * a random QA sample (default 10%) of the auto-accepted items goes too, so
    the pre-labeler's real agreement rate is measured, not assumed.
Annotators fill `decision` (accept / correct / exclude) and, when correcting,
`corrected_lvef`, in a CSV opened in the Workspace's JupyterLab (no extra
install). Report text stays inside the Workspace - never commit these files.
"""
from __future__ import annotations

import csv
import random
from typing import Iterable

from .report_labels import annotate_report, ef_category

COLS = ["study_id", "kind", "reason_in_queue", "auto_lvef", "auto_category", "lvef_span", "flags",
        "report_text", "reviewer", "decision", "corrected_lvef", "notes"]


def export_review_queue(records: Iterable[dict], path: str, qa_fraction: float = 0.10, seed: int = 0) -> dict:
    """records: dicts with study_id, kind ('echo'|'mri'), text."""
    rng = random.Random(seed)
    n = flagged = qa = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in records:
            n += 1
            a = annotate_report(r["text"], r["kind"])
            why = None
            if a["needs_review"]:
                why, flagged = "flagged", flagged + 1
            elif rng.random() < qa_fraction:
                why, qa = "qa_sample", qa + 1
            if not why:
                continue
            lv = a["lvef"]
            w.writerow({"study_id": r["study_id"], "kind": r["kind"], "reason_in_queue": why,
                        "auto_lvef": lv["value"], "auto_category": lv["category"], "lvef_span": lv["span"],
                        "flags": "; ".join(x["finding"] + "=" + x["state"] for x in a["findings"] if x["state"] in ("present", "unclear")),
                        "report_text": r["text"]})
    return {"total": n, "flagged": flagged, "qa_sample": qa}


def apply_reviews(path: str) -> dict:
    """Merge reviewed CSV -> {'labels': {study_id: lvef}, 'excluded': [...], 'pending': [...], 'qa_agreement': float|None}"""
    labels, excluded, pending, qa_total, qa_ok = {}, [], [], 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = (row["decision"] or "").strip().lower()
            sid = row["study_id"]
            if d not in ("accept", "correct", "exclude"):
                pending.append(sid); continue
            if row["reason_in_queue"] == "qa_sample":
                qa_total += 1; qa_ok += d == "accept"
            if d == "exclude":
                excluded.append(sid)
            elif d == "accept":
                if row["auto_lvef"]:
                    labels[sid] = {"lvef": float(row["auto_lvef"]), "category": row["auto_category"]}
                else:
                    pending.append(sid)
            else:
                v = float(row["corrected_lvef"])
                labels[sid] = {"lvef": v, "category": ef_category(v)}
    return {"labels": labels, "excluded": excluded, "pending": pending,
            "qa_agreement": (qa_ok / qa_total) if qa_total else None}
