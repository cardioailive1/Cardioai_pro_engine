"""Run: python -m training.mayo.test_label_audit  (from backend/)"""
import csv, os, tempfile
from training.mayo.label_audit import (auto_label_all, export_ds_audit, read_ds_audits, route,
                                       export_cardiologist_queue, finalize, audit_summary)
from training.mayo.audit_gate import AutoAuditRequired, AuditedLabels
from training.mayo.label_auto_audit import auto_audit


def fill(path, rule):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["auditor"], (r["ds_decision"], r["ds_lvef"]) = "ds1", rule(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def t():
    tmp = tempfile.mkdtemp(); P = lambda n: os.path.join(tmp, n)
    recs = [{"study_id": f"A{i}", "kind": "echo", "text": f"LVEF {40 + i}%."} for i in range(30)]
    recs += [{"study_id": "WRONG", "kind": "echo", "text": "LVEF 55%."},          # DS will disagree
             {"study_id": "RANGE", "kind": "echo", "text": "EF 48-52%."},          # labeler flags -> uncertain
             {"study_id": "UNSURE", "kind": "mri", "text": "LVEF 60%."},          # DS uncertain
             {"study_id": "NONE", "kind": "mri", "text": "Not reported."},        # DS excludes
             {"study_id": "NOAUDIT", "kind": "echo", "text": "LVEF 50%."}]        # never audited
    auto = auto_label_all(recs)
    res = auto_audit(auto)                                     # the automatic audit is required after automatic labeling
    for bad in (None, {}, {k: v for k, v in list(res.items())[:-1]}):          # none, empty or partial audit: every step refuses
        for call in (lambda b=bad: export_ds_audit(auto, P("x.csv"), b), lambda b=bad: route(auto, {}, b)):
            try:
                call(); assert False, "ran without a full automatic audit"
            except AutoAuditRequired:
                pass
    export_ds_audit(auto, P("ds.csv"), res)

    def rule(r):
        s = r["study_id"]
        if s == "NOAUDIT": return ("", "")
        if s == "WRONG": return ("disagree", "35")
        if s == "UNSURE": return ("uncertain", "")
        if s == "NONE": return ("exclude", "")
        return ("agree", "")
    fill(P("ds.csv"), rule)
    audits = read_ds_audits([P("ds.csv")])
    plan = route(auto, audits, res)
    assert plan["NOAUDIT"]["state"] == "pending_ds"
    assert plan["WRONG"]["state"] == "adjudicate" and "disagrees" in plan["WRONG"]["reason"]
    assert plan["RANGE"]["state"] == "adjudicate" and "labeler flagged" in plan["RANGE"]["reason"]
    assert plan["UNSURE"]["state"] == "adjudicate" and plan["NONE"]["state"] == "excluded"
    assert plan["A3"]["state"] == "agreed"
    try:                                   # tier 1 must cover everything
        export_cardiologist_queue(auto, plan, P("c.csv"), P("k.csv")); assert False
    except ValueError:
        pass
    audits["NOAUDIT"] = [{"auditor": "ds1", "decision": "agree", "lvef": None}]
    plan = route(auto, audits, res)
    unaudited_plan = {s: {k: v for k, v in p.items() if k != "auto_audit"} for s, p in plan.items()}   # a plan built without the audit
    for call in (lambda: export_cardiologist_queue(auto, unaudited_plan, P("c0.csv"), P("k0.csv")), lambda: finalize(unaudited_plan, P("c0.csv"), P("k0.csv"))):
        try:
            call(); assert False
        except AutoAuditRequired:
            pass
    st = export_cardiologist_queue(auto, plan, P("c.csv"), P("k.csv"), spot_fraction=0.5, seed=1)
    assert st["adjudication"] == 3 and st["spot_check"] >= 5
    head = open(P("c.csv"), encoding="utf-8").readline()
    assert "auto" not in head and "ds_" not in head and "queue" not in head   # blinded
    key = {r["blind_id"]: r for r in csv.DictReader(open(P("k.csv"), encoding="utf-8"))}
    rows = list(csv.DictReader(open(P("c.csv"), encoding="utf-8")))
    for r in rows:
        k = key[r["blind_id"]]; sid = k["study_id"]
        r["cardiologist_decision"] = "value"
        r["cardiologist_lvef"] = {"WRONG": "36", "RANGE": "50", "UNSURE": "61"}.get(sid, str(plan[sid]["value"]))
    with open(P("c.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    out = finalize(plan, P("c.csv"), P("k.csv"), min_spot_n=5)
    assert out["labels"]["WRONG"]["lvef"] == 36 and out["labels"]["RANGE"]["category"] == "preserved_ge50"
    assert "NONE" in out["excluded"] and not out["unresolved"]
    assert out["spot_check"]["agreement"] == 1.0 and out["passed"]
    assert out["auto_audited"] and isinstance(out["labels"], AuditedLabels) and out["auto_audit_counts"]["studies"] == len(auto)
    assert abs(out["labels"]["A3"]["lvef"] - 43) < 1e-9          # un-spot-checked agreed labels pass through
    # a wrong consensus caught by the blinded spot-check fails the batch
    for r in rows:
        k = key[r["blind_id"]]
        if k["queue_type"] == "spot_check":
            r["cardiologist_lvef"] = "20"; break
    with open(P("c.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    bad = finalize(plan, P("c.csv"), P("k.csv"), min_spot_n=5)
    assert bad["spot_check"]["errors"] and not bad["passed"]
    s = audit_summary(auto, audits)
    assert s["coverage"] == 1.0
    print("all ok")

if __name__ == "__main__":
    t()
