"""
CardioAI Pro - MACE outcome-label review (automatic audit + two-tier human review)
=================================================================================
The MACE outcome labels are built by rules from OMOP (omop_adapter.py). The same
two-tier design as the echo labels (label_audit.py) is applied here, adapted to
what can go wrong with an OUTCOME label:

  * a code was entered for a HISTORY of MI / stroke or a rule-out, not a new event
  * the code's date is the entry date, not the event date, so the 30 / 90-day
    window flips (the window decides the training label)
  * a death with no usable cause is silently labeled 'no event'
  * a prior event just outside the 90-day washout should have excluded the record

  AUTOMATIC AUDIT (auto_audit)   REQUIRED after automatic labeling. Machine checks on each record's evidence, pass/warn/fail.
                                 Every later step refuses to run without a result for every record (audit_gate.py).
  TIER 1  data scientists audit EVERY record (or, opt-in, the non-pass ones plus a sample).
          Decision per record: agree / disagree (corrected event type + days) / uncertain / exclude.
  TIER 2  the cardiologist adjudicates disagreements, uncertain cases, automatic-audit fails,
          and a BLINDED spot-check sampled separately from agreed events and agreed non-events
          (a missed event hidden in a 'no event' label is the costly error). The file is shuffled,
          uses random blind ids, shows only the OMOP evidence (never the auto label, auditor
          decisions or why it was queued); the key stays with the data scientists.

`apply_to_records` returns corrected MACELabelRecords (excluded / unresolved removed) that go
straight into LabelValidator and the training scripts. Files contain patient-level evidence:
keep them inside the Mayo Workspace and never commit them.
"""
from __future__ import annotations

import csv
import math
import random
import secrets
from datetime import timedelta
from typing import Iterable, Optional

from training.label_schema import MACELabelRecord, MACEEventType, MACE_QUALIFYING, compute_window
from training.omop_adapter import OMOPConceptMapping, _to_date
from training.mayo.audit_gate import AuditedLabels, audit_counts, require_audited_labels, require_auto_audit, require_plan_audited
from training.mayo.label_audit import _wilson

NONE = "none"
EVENT_TYPES = {e.value for e in MACE_QUALIFYING}
DAY_TOL = 7           # two reviewers agree on timing if within a week AND in the same label window
BOUNDARY_TOL = 7      # days from the 30 / 90-day boundary that make the window fragile
WASHOUT_TOL = 14

DS_COLS = ["record_id", "auto_event_type", "auto_days_to_event", "auto_window", "auto_audit_status",
           "auto_audit_notes", "evidence", "auditor", "ds_decision", "ds_event_type", "ds_days_to_event", "notes"]
CARD_COLS = ["blind_id", "evidence", "cardiologist_decision", "cardiologist_event_type",
             "cardiologist_days_to_event", "notes"]
KEY_COLS = ["blind_id", "record_id", "queue_type", "reason", "auto_event_type", "auto_days_to_event", "ds_values"]


def _same(a: tuple, b: tuple) -> bool:
    """Outcomes agree: same event type (or both none) and, for events, same window and timing within DAY_TOL."""
    if a[0] != b[0]:
        return False
    if a[0] == NONE:
        return True
    if a[1] is None or b[1] is None or abs(a[1] - b[1]) > DAY_TOL:
        return False
    return _win(a) == _win(b)


def _win(o: tuple) -> str:
    return "none" if o[0] == NONE else ("imminent" if o[1] < 30 else "early" if o[1] < 90 else "late")


def _auto(rec: MACELabelRecord) -> tuple:
    return (rec.event_type.value if rec.event_type else NONE, rec.days_to_event if rec.event_type else None)


# ---------------------------------------------------------------- evidence
def build_review_items(records: Iterable[MACELabelRecord], condition_rows: list[dict], death_rows: list[dict],
                       mapping: OMOPConceptMapping, visit_rows: Optional[list[dict]] = None,
                       lookback_days: int = 365, lookahead_days: int = 365) -> dict:
    """-> {record_id: item}. Evidence = every qualifying MI/stroke condition and any death row near the
    index date, with offsets in days, concept ids/names and (if visit_rows given) the visit setting."""
    visit_type = {v["visit_occurrence_id"]: str(v.get("visit_concept_name", v.get("visit_concept_id", ""))) for v in (visit_rows or [])}
    qual = mapping.mi_condition_concept_ids | mapping.stroke_condition_concept_ids
    by_p_c, by_p_d = {}, {}
    for c in condition_rows:
        if c["condition_concept_id"] in qual:
            by_p_c.setdefault(str(c["person_id"]), []).append(c)
    for d in death_rows:
        by_p_d.setdefault(str(d["person_id"]), []).append(d)
    items = {}
    for r in records:
        if not r.index_date:
            raise ValueError(f"{r.record_id}: no index_date; rebuild records with the current omop_adapter")
        idx = _to_date(r.index_date)
        ev, lines = [], []
        for c in sorted(by_p_c.get(r.patient_id, []), key=lambda c: _to_date(c["condition_start_date"])):
            off = (_to_date(c["condition_start_date"]) - idx).days
            if -lookback_days <= off <= lookahead_days:
                kind = "MI" if c["condition_concept_id"] in mapping.mi_condition_concept_ids else "stroke"
                ev.append({"type": "condition", "kind": kind, "offset": off, "visit": visit_type.get(c.get("visit_occurrence_id"), "")})
                lines.append(f"day {off:+d}: {kind} code {c['condition_concept_id']} {c.get('condition_concept_name', '')}".rstrip()
                             + (f" [{visit_type[c['visit_occurrence_id']]}]" if c.get("visit_occurrence_id") in visit_type else ""))
        for d in by_p_d.get(r.patient_id, []):
            dd = _to_date(d.get("death_date"))
            if dd is not None and 0 <= (dd - idx).days <= lookahead_days:
                off = (dd - idx).days
                cv = mapping.death_counts(d)   # True when this death counts as an outcome event under the chosen definition
                has_cause = d.get("cause_concept_id") not in (None, 0)
                ev.append({"type": "death", "offset": off, "cv": cv, "has_cause": has_cause})
                lines.append(f"day {off:+d}: DEATH, cause {d.get('cause_concept_id') or 'not recorded'}"
                             + (" (counts as an event)" if cv else ""))
        items[r.record_id] = {"record": r, "auto": _auto(r), "evidence": ev,
                              "evidence_text": f"Index ECG date {r.index_date}. " + ("; ".join(lines) if lines else "No MI/stroke code or death near the index date."),
                              "window": r.label_window.value if r.label_window else _win(_auto(r))}
    return items


# ---------------------------------------------------------------- automatic audit
def auto_audit(items: dict) -> dict:
    """-> {record_id: {status, checks:[(name, pass|warn|fail, detail)], suggested}}"""
    out = {}
    for rid, it in items.items():
        et, days = it["auto"]; rec = it["record"]; ev = it["evidence"]
        conds = [e for e in ev if e["type"] == "condition" and e["offset"] >= 0]
        deaths = [e for e in ev if e["type"] == "death"]
        checks, suggested = [], None
        if et == NONE and any(not d["cv"] and not d["has_cause"] for d in deaths):
            checks.append(("death_cause_missing", "fail", "patient died in follow-up with no usable cause; 'no event' may be a missed cardiovascular death"))
        if et != NONE and deaths and any(d["offset"] < days for d in deaths):
            checks.append(("event_after_death", "fail", "event code dated after the death date"))
        if et != NONE and 0 < days and (abs(days - 30) <= BOUNDARY_TOL or abs(days - 90) <= BOUNDARY_TOL):
            checks.append(("borderline_window", "warn", f"event at day {days}; a few days of date error changes the label window"))
        if et in ("myocardial_infarction", "stroke") and sum(1 for c in conds if c["offset"] == days) == 1 \
                and not any(c["offset"] > days for c in conds):
            checks.append(("single_code_support", "warn", "event rests on one code with no repeat or follow-on code"))
        if et in ("myocardial_infarction", "stroke"):
            vis = [c["visit"] for c in conds if c["offset"] == days and c["visit"]]
            if vis and not any(("inpatient" in v.lower() or "emergency" in v.lower() or "er " in v.lower()) for v in vis):
                checks.append(("non_acute_setting", "warn", "event code came from a non-inpatient / non-emergency visit (may be history or billing carry-over)"))
        kinds = {c["kind"] for c in conds if et != NONE and abs(c["offset"] - (days or 0)) <= 7}
        if len(kinds) > 1:
            checks.append(("conflicting_types", "warn", "MI and stroke codes within a week of each other"))
        prior = rec.days_since_prior_mace
        if prior is not None and abs(prior - 90) <= WASHOUT_TOL:
            checks.append(("prior_event_near_washout", "warn", f"prior event {prior} days before index; right at the 90-day washout"))
        if et == NONE and any(c["offset"] >= 0 for c in ev if c["type"] == "condition"):
            checks.append(("code_present_but_no_event", "warn", "MI/stroke code after the index date but the record is labeled no event"))
        status = "fail" if any(c[1] == "fail" for c in checks) else "warn" if checks else "pass"
        out[rid] = {"status": status, "checks": checks or [("all_checks", "pass", "")], "suggested": suggested}
    return out


def triage_ds_scope(items: dict, aa: dict, sample_fraction: float = 0.20, min_sample: int = 10, seed: int = 0) -> set:
    """OPT-IN lighter tier 1: data scientists audit every non-pass record plus a random share of the passes,
    sampled so that events are always included in the sample at a higher share (all passed EVENTS are audited)."""
    rng = random.Random(seed)
    events = {r for r, it in items.items() if it["auto"][0] != NONE}
    clean = sorted(r for r in items if aa[r]["status"] == "pass" and r not in events)
    n = min(len(clean), max(min_sample, math.ceil(sample_fraction * len(clean)))) if clean else 0
    return {r for r in items if aa[r]["status"] != "pass"} | events | set(rng.sample(clean, n))


# ---------------------------------------------------------------- tier 1
def export_ds_audit(items: dict, path: str, aa: dict, ds_scope: Optional[set] = None) -> int:
    require_auto_audit(items, aa, "record")      # the automatic audit is a required step after automatic labeling
    ids = [r for r in items if ds_scope is None or r in ds_scope]
    order = {"fail": 0, "warn": 1, "pass": 2}
    ids.sort(key=lambda r: order[aa[r]["status"]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DS_COLS); w.writeheader()
        for rid in ids:
            it, a = items[rid], (aa or {}).get(rid)
            w.writerow({"record_id": rid, "auto_event_type": it["auto"][0], "auto_days_to_event": it["auto"][1] if it["auto"][1] is not None else "",
                        "auto_window": it["window"], "auto_audit_status": a["status"] if a else "",
                        "auto_audit_notes": "; ".join(f"{c[0]}: {c[2]}" for c in a["checks"] if c[1] != "pass") if a else "",
                        "evidence": it["evidence_text"]})
    return len(ids)


def read_ds_audits(paths: Iterable[str]) -> dict:
    """-> {record_id: [{auditor, decision, event_type, days}]}; blank decisions = not yet audited."""
    out: dict = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = (row["ds_decision"] or "").strip().lower()
                if d not in ("agree", "disagree", "uncertain", "exclude"):
                    continue
                et = (row["ds_event_type"] or "").strip().lower() or None
                days = int(row["ds_days_to_event"]) if (row["ds_days_to_event"] or "").strip() else None
                if d == "disagree":
                    if et not in EVENT_TYPES | {NONE}:
                        raise ValueError(f"{row['record_id']}: 'disagree' needs ds_event_type in {sorted(EVENT_TYPES | {NONE})}")
                    if et != NONE and days is None:
                        raise ValueError(f"{row['record_id']}: an event needs ds_days_to_event")
                out.setdefault(row["record_id"], []).append({"auditor": row["auditor"], "decision": d, "event_type": et, "days": days})
    return out


def route(items: dict, audits: dict, aa: dict, ds_scope: Optional[set] = None) -> dict:
    """state: pending_ds | agreed | adjudicate | excluded. Same rules as the echo design: uncertain or any
    disagreement or an automatic-audit fail goes to the cardiologist; outside an opt-in scope, a record that
    passed every check counts as audited by 'auto-audit' (auto_audit_only)."""
    require_auto_audit(items, aa, "record")      # the automatic audit is a required step after automatic labeling
    plan = {}
    for rid, it in items.items():
        rows, aa_only = audits.get(rid, []), False
        if not rows and ds_scope is not None and rid not in ds_scope and aa and aa[rid]["status"] == "pass" and it["auto"][0] == NONE:
            rows, aa_only = [{"auditor": "auto-audit", "decision": "agree", "event_type": None, "days": None}], True
        if not rows:
            plan[rid] = {"state": "pending_ds", "reason": "no data-scientist audit yet", "value": None, "ds_values": [], "auto_audit_only": False}
            continue
        vals = [it["auto"] if r["decision"] == "agree" else (r["event_type"], r["days"]) for r in rows if r["decision"] in ("agree", "disagree")]
        decisions = {r["decision"] for r in rows}
        reason = None
        if "uncertain" in decisions:
            reason = "uncertain: auditor unsure"
        elif decisions == {"exclude"}:
            plan[rid] = {"state": "excluded", "reason": "all auditors excluded", "value": None, "ds_values": [], "auto_audit_only": aa_only}
            continue
        elif "exclude" in decisions:
            reason = "disagreement: one auditor excluded, another gave an outcome"
        elif "disagree" in decisions:
            reason = "disagreement: auditor disagrees with auto label"
        elif any(not _same(vals[0], v) for v in vals[1:]):
            reason = "disagreement: auditors differ"
        elif aa and aa[rid]["status"] == "fail":
            reason = "disagreement: automatic audit failed (" + ", ".join(c[0] for c in aa[rid]["checks"] if c[1] == "fail") + ")"
        if reason:
            plan[rid] = {"state": "adjudicate", "reason": reason, "value": None, "ds_values": vals, "auto_audit_only": aa_only}
        else:
            plan[rid] = {"state": "agreed", "reason": "", "value": vals[0], "ds_values": vals, "auto_audit_only": aa_only}
    for rid in plan:
        plan[rid]["auto_audit"] = aa[rid]["status"]    # recorded so later steps can verify the audit was run
    return plan


# ---------------------------------------------------------------- tier 2
def export_cardiologist_queue(items: dict, plan: dict, path: str, key_path: str, spot_fraction: float = 0.10,
                              min_spot: int = 5, seed: int = 0, auto_only_spot_fraction: float = 0.15) -> dict:
    """Blinded queue: all 'adjudicate' records + spot-checks sampled SEPARATELY from agreed events and agreed
    non-events (and at a higher rate for records that skipped the human audit)."""
    require_plan_audited(plan, "record")
    pending = [r for r, p in plan.items() if p["state"] == "pending_ds"]
    if pending:
        raise ValueError(f"{len(pending)} records not yet audited by data scientists (e.g. {pending[:3]})")
    rng = random.Random(seed)
    def pick(pool, frac):
        pool = sorted(pool)
        return set(rng.sample(pool, min(len(pool), max(min_spot, math.ceil(frac * len(pool)))))) if pool else set()
    agreed = [r for r, p in plan.items() if p["state"] == "agreed" and not p["auto_audit_only"]]
    spot = pick([r for r in agreed if plan[r]["value"][0] != NONE], spot_fraction) | pick([r for r in agreed if plan[r]["value"][0] == NONE], spot_fraction)
    spot |= pick([r for r, p in plan.items() if p["state"] == "agreed" and p["auto_audit_only"]], auto_only_spot_fraction)
    queue = [(r, "adjudication") for r, p in plan.items() if p["state"] == "adjudicate"] + [(r, "spot_check") for r in spot]
    rng.shuffle(queue)
    with open(path, "w", newline="", encoding="utf-8") as f, open(key_path, "w", newline="", encoding="utf-8") as k:
        w = csv.DictWriter(f, fieldnames=CARD_COLS); w.writeheader()
        kw = csv.DictWriter(k, fieldnames=KEY_COLS); kw.writeheader()
        for rid, qt in queue:
            bid = secrets.token_hex(4)
            w.writerow({"blind_id": bid, "evidence": items[rid]["evidence_text"]})
            kw.writerow({"blind_id": bid, "record_id": rid, "queue_type": qt,
                         "reason": plan[rid]["reason"] or ("auto-audit-only" if plan[rid]["auto_audit_only"] else ""),
                         "auto_event_type": items[rid]["auto"][0], "auto_days_to_event": items[rid]["auto"][1] or "",
                         "ds_values": ";".join(f"{v[0]}@{v[1]}" for v in plan[rid]["ds_values"])})
    return {"adjudication": sum(1 for _, q in queue if q == "adjudication"), "spot_check": len(spot)}


def finalize(plan: dict, cardio_path: str, key_path: str, target_agreement: float = 0.95, min_spot_n: int = 20) -> dict:
    """Merge cardiologist decisions (cardiologist > consensus). Reports spot-check agreement overall and
    separately for events and non-events. Passes at >= target agreement over >= min_spot_n with nothing unresolved."""
    require_plan_audited(plan, "record")
    key = {r["blind_id"]: r for r in csv.DictReader(open(key_path, newline="", encoding="utf-8"))}
    answers = {}
    for r in csv.DictReader(open(cardio_path, newline="", encoding="utf-8")):
        d = (r["cardiologist_decision"] or "").strip().lower()
        if d in ("event", "none", "exclude"):
            et = r["cardiologist_event_type"].strip().lower() if d == "event" else NONE
            if d == "event" and et not in EVENT_TYPES:
                raise ValueError(f"{r['blind_id']}: event needs cardiologist_event_type in {sorted(EVENT_TYPES)}")
            days = int(r["cardiologist_days_to_event"]) if d == "event" else None
            if d == "event" and days is None:
                raise ValueError(f"{r['blind_id']}: event needs cardiologist_days_to_event")
            answers[r["blind_id"]] = ("exclude", None) if d == "exclude" else (et, days)
    final, excluded, unresolved = {}, [], []
    st = {"event": [0, 0], "none": [0, 0]}
    errors = []
    for bid, k in key.items():
        rid, ans = k["record_id"], answers.get(bid)
        if ans is None:
            unresolved.append(rid); continue
        if k["queue_type"] == "spot_check":
            cons = plan[rid]["value"]; good = ans[0] != "exclude" and _same(cons, ans)
            grp = "none" if cons[0] == NONE else "event"
            st[grp][0] += 1; st[grp][1] += good
            if not good:
                errors.append(rid)
        if ans[0] == "exclude":
            excluded.append(rid)
        else:
            final[rid] = ans
    queued = {k["record_id"] for k in key.values()}
    for rid, p in plan.items():
        if rid in queued:
            continue
        if p["state"] == "agreed":
            final[rid] = p["value"]
        elif p["state"] == "excluded":
            excluded.append(rid)
        else:
            unresolved.append(rid)
    unresolved = sorted(set(unresolved))
    for rid in unresolved:
        final.pop(rid, None)
    n = st["event"][0] + st["none"][0]; ok = st["event"][1] + st["none"][1]
    agree = ok / n if n else None
    return {"outcomes": AuditedLabels(final), "auto_audited": True, "auto_audit_counts": audit_counts(plan), "excluded": sorted(set(excluded)), "unresolved": unresolved,
            "spot_check": {"n": n, "agree": ok, "agreement": agree, "wilson95": _wilson(ok, n), "errors": errors,
                           "events": {"n": st["event"][0], "agree": st["event"][1]},
                           "non_events": {"n": st["none"][0], "agree": st["none"][1]}},
            "passed": bool(agree is not None and n >= min_spot_n and agree >= target_agreement and not unresolved),
            "note": "spot-check too small to certify" if n < min_spot_n else ""}


def apply_to_records(records: Iterable[MACELabelRecord], result: dict) -> list[MACELabelRecord]:
    """Corrected records for LabelValidator / training: reviewed outcomes applied, excluded and unresolved dropped."""
    require_audited_labels(result.get("outcomes"))
    out = []
    for r in records:
        o = result["outcomes"].get(r.record_id)
        if o is None:
            continue
        et, days = o
        r.event_type = MACEEventType(et) if et != NONE else None
        r.days_to_event = days
        r.censored = et == NONE   # a corrected 'none' keeps its observed follow-up; LabelValidator drops it if too short
        if et != NONE:
            r.follow_up_days = max(r.follow_up_days, days)
        r.label_window = compute_window(r)
        out.append(r)
    return out
