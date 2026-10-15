"""
CardioAI Pro - automatic label audit (optional, runs before the cardiologist)
============================================================================
An independent set of machine checks on each auto label, to help the data
scientists audit faster and to catch the errors a single regex pass makes. It
does NOT replace the data-scientist audit by default (see label_audit.route):
by default the results are shown to the auditors as extra columns. Only if you
opt in (label_audit.triage_ds_scope) do clean studies get a lighter human
audit - and those are then oversampled in the blinded cardiologist spot-check.

Checks (status pass / warn / fail):
  value_present       a numeric LVEF was found
  plausible_range     10-80 %
  prior_context       the matched EF sits next to "prior / previous / compared with /
                      baseline / post-procedure" wording (labeler takes the first
                      EF it finds - this catches picking a historical value)
  multiple_values     different EF values in one report
  qualitative_match   words like "normal", "mildly/moderately/severely reduced"
                      agree with the number (ranges are heuristics, adjustable)
  impression_match    EF in RADIOLOGY_IMPRESSION agrees with the report body (optional input)
  structured_match    a structured EF measurement (if the record has `structured_lvef`) agrees
  duplicate_conflict  identical report text elsewhere in the batch got a different label
A 'fail' may carry suggested_lvef (from the impression / structured value).
"""
from __future__ import annotations

import re
from typing import Optional

from .report_labels import extract_lvef, _EF

TOL = 5.0
# (regex, low, high) heuristic LVEF ranges for qualitative wording - clinical teams should confirm
QUAL = [
    (re.compile(r"\b(?:severely|markedly)\s+(?:reduced|decreased|depressed|impaired)", re.I), 0, 36),
    (re.compile(r"\bmoderately\s+(?:reduced|decreased|depressed|impaired)", re.I), 28, 46),
    (re.compile(r"\bmildly\s+(?:reduced|decreased|depressed|impaired)", re.I), 38, 56),
    (re.compile(r"\b(?:normal|preserved)\s+(?:global\s+)?(?:lv|left ventricular|systolic)\s+(?:function|ejection)", re.I), 50, 80),
]
_PRIOR = re.compile(r"\b(prior|previous|previously|compared (?:to|with)|baseline|historical|post[- ]?(?:op|procedure|ablation|mi)|was)\b", re.I)


def _close(a, b):
    return a is not None and b is not None and abs(a - b) <= TOL


def auto_audit(auto: dict) -> dict:
    """auto: label_audit.auto_label_all output (may carry impression / structured_lvef).
    -> {study_id: {status, checks: [[name, status, detail]], suggested_lvef}}"""
    by_text: dict = {}
    for sid, a in auto.items():
        by_text.setdefault(re.sub(r"\s+", " ", a["text"].strip().lower()), []).append(sid)
    out = {}
    for sid, a in auto.items():
        text, v, checks, sugg = a["text"], a["lvef"], [], None

        def add(name, st, detail=""):
            checks.append([name, st, detail])
        add("value_present", "pass" if v is not None else "fail", "" if v is not None else "no EF value found")
        if v is not None:
            add("plausible_range", "pass" if 10 <= v <= 80 else "fail", f"{v}%")
            ms = list(_EF.finditer(text))
            if ms:
                m = ms[0]
                ctx = text[max(0, m.start() - 60):m.start()]
                add("prior_context", "warn" if _PRIOR.search(ctx) else "pass", ctx.strip()[-60:] if _PRIOR.search(ctx) else "")
            vals = {(float(m.group(1)), float(m.group(2)) if m.group(2) else None) for m in ms}
            add("multiple_values", "warn" if len(vals) > 1 else "pass", f"{len(vals)} distinct EF values" if len(vals) > 1 else "")
            hit = next(((lo, hi) for rx, lo, hi in QUAL if rx.search(text)), None)
            if hit:
                add("qualitative_match", "pass" if hit[0] <= v <= hit[1] else "fail",
                    f"wording implies {hit[0]}-{hit[1]}%, label {v}%")
        imp = a.get("impression")
        if imp:
            iv = extract_lvef(imp).value
            if iv is not None and v is not None:
                ok = _close(v, iv)
                add("impression_match", "pass" if ok else "fail", f"impression {iv}%")
                if not ok:
                    sugg = iv
            elif iv is not None:
                add("impression_match", "warn", f"impression has {iv}% but body label is empty"); sugg = iv
        sv = a.get("structured_lvef")
        if sv is not None:
            ok = _close(v, float(sv))
            add("structured_match", "pass" if ok else "fail", f"structured {sv}%")
            if not ok:
                sugg = float(sv)
        peers = by_text[re.sub(r"\s+", " ", text.strip().lower())]
        if len(peers) > 1:
            others = {auto[p]["lvef"] for p in peers if p != sid}
            if any(not _close(v, o) for o in others):
                add("duplicate_conflict", "fail", f"identical text labeled differently ({len(peers)} copies)")
        statuses = {c[1] for c in checks}
        out[sid] = {"status": "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass"),
                    "checks": checks, "suggested_lvef": sugg}
    return out


def summarize(results: dict) -> dict:
    n = len(results)
    c = {s: sum(1 for r in results.values() if r["status"] == s) for s in ("pass", "warn", "fail")}
    return {"studies": n, **c, "pass_rate": (c["pass"] / n) if n else None}
