"""
CardioAI Pro - two-tier label audit
===================================
Automatic labels (report_labels.py) are audited before any training use. The AUTOMATIC AUDIT
(label_auto_audit.py) is a required first step: every study must have a result, and each later step
refuses to run without it (audit_gate.py).

  TIER 1 - data scientists audit EVERYTHING.
      Every auto-labeled study goes to at least one data-scientist auditor
      (export_ds_audit). Each marks agree / disagree (with their LVEF) /
      uncertain / exclude. Finalizing refuses to proceed past studies no
      auditor has covered.

  TIER 2 - the cardiologist adjudicates only:
      * disagreements (an auditor disagrees with the auto label, or two
        auditors disagree with each other),
      * uncertain cases (an auditor unsure, or the labeler itself flagged
        needs_review - ranges, prior-vs-current EF, addenda...),
      * a small BLINDED spot-check of studies everyone agreed on.
      The cardiologist's file is shuffled, uses random blind ids, and never
      shows the auto label, auditor decisions or why a study was queued -
      so spot-check items cannot be told apart from adjudication items, and
      no one's answer anchors the clinician. Sources are kept in a separate
      key file that stays with the data scientists.

Output of `finalize` is {study_id: {lvef, category}} - the same shape as
review_queue.apply_reviews, so it feeds lvef_dataset.build_lvef_examples
unchanged. Also reports auditor-vs-auto agreement and the spot-check
agreement of the consensus label against the cardiologist (the real
error-rate estimate). All files contain report text: keep them inside the
Mayo Workspace, never commit them.
"""
from __future__ import annotations

import csv
import math
import random
import secrets
from typing import Iterable, Optional

from .audit_gate import AuditedLabels, audit_counts, require_auto_audit, require_plan_audited
from .report_labels import annotate_report, ef_category

TOL = 5.0  # LVEF points; two values within TOL AND in the same HF category "agree"

DS_COLS = ["study_id", "kind", "auto_lvef", "auto_category", "labeler_flagged",
           "auto_audit_status", "auto_audit_notes", "auto_audit_suggested_lvef", "report_text",
           "auditor", "ds_decision", "ds_lvef", "notes"]
CARD_COLS = ["blind_id", "kind", "report_text", "candidate_lvef_values", "cardiologist_decision",
             "cardiologist_lvef", "notes"]
KEY_COLS = ["blind_id", "study_id", "queue_type", "reason", "auto_lvef", "ds_values"]


def _same(a: Optional[float], b: Optional[float], tol: float = TOL) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol and ef_category(a) == ef_category(b)


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# ---------------------------------------------------------------- tier 1
def auto_label_all(records: Iterable[dict]) -> dict:
    """records: {study_id, kind, text}. -> {study_id: {kind, text, lvef, category, flagged}}"""
    out = {}
    for r in records:
        a = annotate_report(r["text"], r["kind"])
        out[r["study_id"]] = {"kind": r["kind"], "text": r["text"], "lvef": a["lvef"]["value"],
                              "category": a["lvef"]["category"], "flagged": bool(a["needs_review"]),
                              "impression": r.get("impression"), "structured_lvef": r.get("structured_lvef")}
    return out


def triage_ds_scope(auto: dict, auto_audit: dict, sample_fraction: float = 0.20, min_sample: int = 10,
                    seed: int = 0) -> set:
    """OPT-IN lighter tier 1: data scientists audit every study the automatic audit did not pass, plus a
    random `sample_fraction` of the ones it passed. Pass the result as `ds_scope` to export_ds_audit and
    route. Default behaviour (no scope) is: data scientists audit everything."""
    rng = random.Random(seed)
    clean = sorted(s for s in auto if auto_audit[s]["status"] == "pass")
    n = min(len(clean), max(min_sample, math.ceil(sample_fraction * len(clean)))) if clean else 0
    return {s for s in auto if auto_audit[s]["status"] != "pass"} | set(rng.sample(clean, n))


def export_ds_audit(auto: dict, path: str, auto_audit: dict, ds_scope: Optional[set] = None) -> int:
    """Studies for the data-scientist auditors (give each a copy; fill `auditor`). Every study by default;
    with `ds_scope` only those. With `auto_audit` results, adds auto_audit_status / notes / suggested LVEF
    columns and lists fails first so the auditors see the risky ones early."""
    require_auto_audit(auto, auto_audit)          # the automatic audit is a required step after automatic labeling
    ids = [s for s in auto if ds_scope is None or s in ds_scope]
    order = {"fail": 0, "warn": 1, "pass": 2}
    ids.sort(key=lambda s: order[auto_audit[s]["status"]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DS_COLS); w.writeheader()
        for sid in ids:
            a, r = auto[sid], (auto_audit or {}).get(sid)
            w.writerow({"study_id": sid, "kind": a["kind"], "auto_lvef": a["lvef"], "auto_category": a["category"],
                        "labeler_flagged": a["flagged"],
                        "auto_audit_status": r["status"] if r else "",
                        "auto_audit_notes": "; ".join(f"{c[0]}={c[1]}" + (f" ({c[2]})" if c[2] else "") for c in r["checks"] if c[1] != "pass") if r else "",
                        "auto_audit_suggested_lvef": (r["suggested_lvef"] if r and r["suggested_lvef"] is not None else ""),
                        "report_text": a["text"]})
    return len(ids)


def read_ds_audits(paths: Iterable[str]) -> dict:
    """-> {study_id: [ {auditor, decision, lvef} ... ]}  (blank decisions ignored = not yet audited)"""
    out: dict = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = (row["ds_decision"] or "").strip().lower()
                if d not in ("agree", "disagree", "uncertain", "exclude"):
                    continue
                v = float(row["ds_lvef"]) if (row["ds_lvef"] or "").strip() else None
                if d == "disagree" and v is None:
                    raise ValueError(f"{row['study_id']}: 'disagree' needs ds_lvef")
                out.setdefault(row["study_id"], []).append({"auditor": row["auditor"], "decision": d, "lvef": v})
    return out


# ---------------------------------------------------------------- routing
def route(auto: dict, audits: dict, auto_audit: dict, ds_scope: Optional[set] = None) -> dict:
    """Decide each study's path. -> {study_id: {state, reason, value, ds_values, auto_audit_only}}
    state: pending_ds | agreed | adjudicate | excluded
    Opt-in triage: with ds_scope + auto_audit, a study OUTSIDE the scope that passed every automatic check
    counts as audited by 'auto-audit' (auto_audit_only=True). Anything else still needs a human audit.
    An automatic 'fail' always sends the study to the cardiologist, even if a data scientist agreed."""
    require_auto_audit(auto, auto_audit)          # the automatic audit is a required step after automatic labeling
    plan = {}
    for sid, a in auto.items():
        rows = audits.get(sid, [])
        aa_only = False
        if not rows and ds_scope is not None and sid not in ds_scope and auto_audit and auto_audit[sid]["status"] == "pass":
            rows, aa_only = [{"auditor": "auto-audit", "decision": "agree", "lvef": None}], True
        if not rows:
            plan[sid] = {"state": "pending_ds", "reason": "no data-scientist audit yet", "value": None, "ds_values": [],
                         "auto_audit_only": False}
            continue
        vals = [a["lvef"] if r["decision"] == "agree" else r["lvef"] for r in rows]
        ds_vals = [v for v in vals if v is not None]
        decisions = {r["decision"] for r in rows}
        reason = None
        if "uncertain" in decisions:
            reason = "uncertain: auditor unsure"
        elif decisions == {"exclude"}:
            plan[sid] = {"state": "excluded", "reason": "all auditors excluded", "value": None, "ds_values": [],
                         "auto_audit_only": aa_only}
            continue
        elif "exclude" in decisions:
            reason = "disagreement: one auditor excluded, another gave a value"
        elif any(r["decision"] == "disagree" for r in rows):
            reason = "disagreement: auditor disagrees with auto label"
        elif any(not _same(ds_vals[0], v) for v in ds_vals[1:]):
            reason = "disagreement: auditors differ"
        elif a["flagged"] or a["lvef"] is None:
            reason = "uncertain: labeler flagged for review"
        elif auto_audit and auto_audit[sid]["status"] == "fail":
            reason = "disagreement: automatic audit failed (" + ", ".join(c[0] for c in auto_audit[sid]["checks"] if c[1] == "fail") + ")"
        if reason:
            plan[sid] = {"state": "adjudicate", "reason": reason, "value": None, "ds_values": ds_vals,
                         "auto_audit_only": aa_only}
        else:
            plan[sid] = {"state": "agreed", "reason": "", "value": sum(ds_vals) / len(ds_vals), "ds_values": ds_vals,
                         "auto_audit_only": aa_only}
    for sid in plan:
        plan[sid]["auto_audit"] = auto_audit[sid]["status"]    # recorded so later steps can verify the audit was run
    return plan


# ---------------------------------------------------------------- tier 2
def export_cardiologist_queue(auto: dict, plan: dict, path: str, key_path: str,
                              spot_fraction: float = 0.05, min_spot: int = 5, seed: int = 0,
                              auto_only_spot_fraction: float = 0.10) -> dict:
    """Blinded, shuffled queue: all 'adjudicate' studies + a random spot-check of 'agreed' ones.
    Refuses if any study still lacks a data-scientist audit (tier 1 must cover everything)."""
    require_plan_audited(plan)
    pending = [s for s, p in plan.items() if p["state"] == "pending_ds"]
    if pending:
        raise ValueError(f"{len(pending)} studies not yet audited by data scientists (e.g. {pending[:3]})")
    rng = random.Random(seed)
    agreed = sorted(s for s, p in plan.items() if p["state"] == "agreed" and not p.get("auto_audit_only"))
    n_spot = min(len(agreed), max(min_spot, math.ceil(spot_fraction * len(agreed)))) if agreed else 0
    spot = set(rng.sample(agreed, n_spot))
    # studies that skipped the human audit (opt-in triage) are checked at a higher rate
    aa = sorted(s for s, p in plan.items() if p["state"] == "agreed" and p.get("auto_audit_only"))
    if aa:
        spot |= set(rng.sample(aa, min(len(aa), max(min_spot, math.ceil(auto_only_spot_fraction * len(aa))))))
    n_spot = len(spot)
    items = [(s, "adjudication") for s, p in plan.items() if p["state"] == "adjudicate"] + [(s, "spot_check") for s in spot]
    rng.shuffle(items)
    with open(path, "w", newline="", encoding="utf-8") as f, open(key_path, "w", newline="", encoding="utf-8") as k:
        w = csv.DictWriter(f, fieldnames=CARD_COLS); w.writeheader()
        kw = csv.DictWriter(k, fieldnames=KEY_COLS); kw.writeheader()
        for sid, qt in items:
            bid = secrets.token_hex(4)
            cands = sorted({round(v, 1) for v in ([auto[sid]["lvef"]] + plan[sid]["ds_values"]) if v is not None})
            w.writerow({"blind_id": bid, "kind": auto[sid]["kind"], "report_text": auto[sid]["text"],
                        "candidate_lvef_values": "; ".join(str(c) for c in cands) if qt == "adjudication" else ""})
            kw.writerow({"blind_id": bid, "study_id": sid, "queue_type": qt, "reason": plan[sid]["reason"] or ("auto-audit-only" if plan[sid].get("auto_audit_only") else ""),
                         "auto_lvef": auto[sid]["lvef"], "ds_values": ";".join(str(v) for v in plan[sid]["ds_values"])})
    return {"adjudication": sum(1 for _, q in items if q == "adjudication"), "spot_check": n_spot}


def finalize(plan: dict, cardio_path: str, key_path: str, target_agreement: float = 0.95, min_spot_n: int = 20) -> dict:
    """Merge cardiologist decisions. Precedence: cardiologist > agreed consensus. Reports coverage,
    auditor/spot-check agreement, and whether the batch passes (spot-check agreement >= target with
    enough samples)."""
    require_plan_audited(plan)
    key = {}
    with open(key_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key[r["blind_id"]] = r
    final, excluded, unresolved = {}, [], []
    spot_n = spot_ok = 0
    spot_errors = []
    aa_n = aa_ok = 0
    answered = {}
    with open(cardio_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r["cardiologist_decision"] or "").strip().lower()
            if d in ("value", "exclude"):
                answered[r["blind_id"]] = (d, r["cardiologist_lvef"])
    for bid, k in key.items():
        sid = k["study_id"]
        ans = answered.get(bid)
        if ans is None:
            unresolved.append(sid); continue
        d, v = ans
        if d == "value":
            v = float(v)
        if k["queue_type"] == "spot_check":
            spot_n += 1
            cons = plan[sid]["value"]
            good = d == "value" and _same(cons, v)
            spot_ok += good
            if not good:
                spot_errors.append(sid)
            if k["reason"] == "auto-audit-only":
                aa_n += 1; aa_ok += good
        if d == "exclude":
            excluded.append(sid)
        else:
            final[sid] = {"lvef": v, "category": ef_category(v)}
    queued = {k["study_id"] for k in key.values()}
    for sid, p in plan.items():
        if sid in queued:
            continue
        if p["state"] == "agreed":
            final[sid] = {"lvef": p["value"], "category": ef_category(p["value"])}
        elif p["state"] == "excluded":
            excluded.append(sid)
        elif p["state"] == "adjudicate":
            unresolved.append(sid)
        else:
            unresolved.append(sid)
    unresolved = sorted(set(unresolved))
    for sid in unresolved:
        final.pop(sid, None)
    agree = spot_ok / spot_n if spot_n else None
    return {"labels": AuditedLabels(final), "auto_audited": True, "auto_audit_counts": audit_counts(plan),
            "excluded": sorted(set(excluded)), "unresolved": unresolved,
            "spot_check": {"n": spot_n, "agree": spot_ok, "agreement": agree, "wilson95": _wilson(spot_ok, spot_n),
                           "errors": spot_errors},
            "spot_check_auto_audit_only": {"n": aa_n, "agree": aa_ok, "agreement": (aa_ok / aa_n) if aa_n else None},
            "passed": bool(agree is not None and spot_n >= min_spot_n and agree >= target_agreement and not unresolved),
            "note": ("spot-check too small to certify" if spot_n < min_spot_n else "")}


def audit_summary(auto: dict, audits: dict) -> dict:
    """Tier-1 stats: coverage and how often auditors agreed with the auto label."""
    cov = sum(1 for s in auto if audits.get(s))
    rows = [r for s in auto for r in audits.get(s, [])]
    agree = sum(1 for r in rows if r["decision"] == "agree")
    return {"studies": len(auto), "audited": cov, "coverage": cov / len(auto) if auto else None,
            "auditor_agree_rate": agree / len(rows) if rows else None}
