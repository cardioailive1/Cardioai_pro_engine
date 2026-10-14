"""
CardioAI Pro - automatic annotation from reports and notes
==========================================================
Mayo confirmed (2026-10-14): the platform holds NO image segmentations or
curated labels. Labels are what we extract from radiology reports
(FACT_RADIOLOGY) and echo notes (FACT_ECHO_TEST.ECHO_NOTE). This module is
that first automatic-annotation pass: rule-based, deterministic, and
auditable. It is a pre-labeler, not ground truth:

  * every label carries the matched text span as provenance;
  * anything ambiguous (several different EF values, ranges, negation
    doubts, no value) is flagged needs_review for the human annotators
    (up to 5 extra seats, cardioailive.com accounts, hired outside Mayo);
  * a labeled study is only trusted for training after review sampling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

_NUM = r"(\d{1,2}(?:\.\d+)?)"
_EF = re.compile(
    r"(?:\bLVEF\b|\bLV\s+EF\b|\bEF\b|ejection\s+fraction)"
    r"[^0-9.\n]{0,40}?" + _NUM + r"\s*(?:-|–|to)?\s*" + r"(?:" + _NUM + r")?\s*%",
    re.I)
_NEG = re.compile(r"\b(no|not|without|negative for|absence of|cannot|unable to|no evidence of)\b", re.I)


@dataclass
class EFLabel:
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    category: Optional[str] = None
    span: str = ""
    needs_review: bool = True
    reason: str = ""


def ef_category(ef: float) -> str:
    """HF phenotype by LVEF: <=40 HFrEF, 41-49 HFmrEF, >=50 HFpEF-range."""
    return "reduced_le40" if ef <= 40 else ("mildly_reduced_41_49" if ef < 50 else "preserved_ge50")


def extract_lvef(text: str) -> EFLabel:
    """Extract LVEF % from an echo note or MRI report."""
    hits = []
    for m in _EF.finditer(text or ""):
        lo = float(m.group(1)); hi = float(m.group(2)) if m.group(2) else None
        if hi is not None and hi < lo:       # '55 - 60' style captured oddly
            hi = None
        if 5 <= lo <= 85 and (hi is None or hi <= 85):
            hits.append((lo, hi, m.group(0).strip()))
    if not hits:
        return EFLabel(reason="no EF value found")
    vals = {(lo, hi) for lo, hi, _ in hits}
    lo, hi, span = hits[0]
    mid = (lo + hi) / 2 if hi is not None else lo
    label = EFLabel(value=mid, low=lo, high=hi if hi is not None else lo, category=ef_category(mid), span=span)
    if len(vals) > 1:
        label.needs_review, label.reason = True, "multiple different EF values (e.g. prior vs current)"
    elif hi is not None and ef_category(lo) != ef_category(hi):
        label.needs_review, label.reason = True, "range straddles a category boundary"
    else:
        label.needs_review, label.reason = False, ""
    return label


def extract_finding(text: str, finding: str, terms: list[str]) -> dict:
    """Present / absent / unclear (conflicting) / not_mentioned for a named finding, negation-aware per
    sentence. terms are lowercase phrases, e.g. ['late gadolinium enhancement','lge']."""
    present = absent = False
    spans = []
    for sent in re.split(r"(?<=[.;])\s+|\n", text or ""):
        low = sent.lower()
        if any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in terms):
            spans.append(sent.strip())
            if _NEG.search(sent): absent = True
            else: present = True
    if not spans:
        state = "not_mentioned"
    else:
        state = "unclear" if (present and absent) else ("present" if present else "absent")
    return {"finding": finding, "state": state, "spans": spans[:3], "needs_review": state == "unclear"}


MRI_FINDINGS = {
    "late_gadolinium_enhancement": ["late gadolinium enhancement", "lge"],
    "myocardial_infarct": ["infarct", "scar"],
    "myocardial_edema": ["edema", "oedema", "t2 hyperintens"],
    "pericardial_effusion": ["pericardial effusion"],
}
ECHO_FINDINGS = {
    "lv_hypertrophy": ["hypertrophy", "lvh"],
    "mitral_regurgitation_moderate_plus": ["moderate mitral regurgitation", "severe mitral regurgitation"],
    "aortic_stenosis": ["aortic stenosis"],
    "regional_wall_motion_abnormality": ["wall motion abnormality", "hypokinesis", "akinesis"],
}


def annotate_report(text: str, kind: str) -> dict:
    """kind: 'echo' (FACT_ECHO_TEST.ECHO_NOTE) or 'mri' (FACT_RADIOLOGY)."""
    table = ECHO_FINDINGS if kind == "echo" else MRI_FINDINGS
    ef = extract_lvef(text)
    findings = [extract_finding(text, k, v) for k, v in table.items()]
    return {"kind": kind, "lvef": asdict(ef), "findings": findings,
            "needs_review": ef.needs_review or any(f["needs_review"] for f in findings)}
