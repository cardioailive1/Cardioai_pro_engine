"""Run: python -m training.mayo.test_label_auto_audit  (from backend/)"""
import csv, os, tempfile
from training.mayo.label_audit import (auto_label_all, export_ds_audit, read_ds_audits, route, triage_ds_scope,
                                       export_cardiologist_queue, finalize)
from training.mayo.label_auto_audit import auto_audit, summarize


def t():
    recs = [{"study_id": f"A{i}", "kind": "echo", "text": f"Normal left ventricular function. LVEF {55 + i % 5}%."} for i in range(40)]
    recs += [
        {"study_id": "PRIOR", "kind": "echo", "text": "Compared with the prior study, EF 35%. Current LVEF 55%."},
        {"study_id": "QUAL", "kind": "echo", "text": "Severely reduced LV function. LVEF 58%."},
        {"study_id": "IMP", "kind": "mri", "text": "LVEF 55%.", "impression": "Mildly reduced EF 42%."},
        {"study_id": "STRUCT", "kind": "echo", "text": "LVEF 60%.", "structured_lvef": 45},
        {"study_id": "DUP1", "kind": "echo", "text": "EF is 50%."}, {"study_id": "DUP2", "kind": "echo", "text": "EF is 50%."},
        {"study_id": "NONE", "kind": "mri", "text": "Not reported."},
    ]
    auto = auto_label_all(recs)
    res = auto_audit(auto)
    st = lambda s, name: {c[0]: c[1] for c in res[s]["checks"]}.get(name)
    assert res["A3"]["status"] == "pass"
    assert st("PRIOR", "prior_context") == "warn" and st("PRIOR", "multiple_values") == "warn"
    assert st("QUAL", "qualitative_match") == "fail" and res["QUAL"]["status"] == "fail"
    assert st("IMP", "impression_match") == "fail" and res["IMP"]["suggested_lvef"] == 42
    assert st("STRUCT", "structured_match") == "fail" and res["STRUCT"]["suggested_lvef"] == 45
    assert res["NONE"]["status"] == "fail"
    assert summarize(res)["pass"] >= 40

    tmp = tempfile.mkdtemp(); P = lambda n: os.path.join(tmp, n)
    # default: everything goes to the data scientists, riskiest first, with the auto-audit columns
    n = export_ds_audit(auto, P("all.csv"), auto_audit=res)
    assert n == len(auto)
    rows = list(csv.DictReader(open(P("all.csv"), encoding="utf-8")))
    assert rows[0]["auto_audit_status"] == "fail" and rows[-1]["auto_audit_status"] == "pass"
    assert any(r["study_id"] == "IMP" and float(r["auto_audit_suggested_lvef"] or 0) == 42 for r in rows)

    # opt-in triage: only non-pass + a sample of passes need a human audit
    scope = triage_ds_scope(auto, res, sample_fraction=0.25, min_sample=5, seed=1)
    assert {"PRIOR", "QUAL", "IMP", "STRUCT", "NONE"} <= scope and 5 <= len(scope & {f"A{i}" for i in range(40)}) <= 15
    export_ds_audit(auto, P("scope.csv"), auto_audit=res, ds_scope=scope)
    rows = list(csv.DictReader(open(P("scope.csv"), encoding="utf-8")))
    assert len(rows) == len(scope)
    for r in rows:
        s = r["study_id"]
        r["auditor"] = "ds1"
        r["ds_decision"], r["ds_lvef"] = (("exclude", "") if s == "NONE" else ("agree", ""))
    with open(P("scope.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    audits = read_ds_audits([P("scope.csv")])
    plan = route(auto, audits, auto_audit=res, ds_scope=scope)
    assert not [s for s, p in plan.items() if p["state"] == "pending_ds"]          # nobody uncovered
    skipped = [s for s in auto if s not in scope]
    assert skipped and all(plan[s]["state"] == "agreed" and plan[s]["auto_audit_only"] for s in skipped)
    for s in ("QUAL", "IMP", "STRUCT"):                                            # DS agreed, auto-audit failed -> cardiologist
        assert plan[s]["state"] == "adjudicate" and "automatic audit failed" in plan[s]["reason"]
    # without opt-in, studies outside scope are NOT auto-covered
    plain = route(auto, audits, res)
    assert any(p["state"] == "pending_ds" for p in plain.values())
    # auto-audit-only studies are oversampled in the blinded spot-check and tracked separately
    q = export_cardiologist_queue(auto, plan, P("c.csv"), P("k.csv"), spot_fraction=0.2, min_spot=2, seed=3,
                                  auto_only_spot_fraction=0.5)
    key = {r["blind_id"]: r for r in csv.DictReader(open(P("k.csv"), encoding="utf-8"))}
    assert any(k["reason"] == "auto-audit-only" for k in key.values())
    crows = list(csv.DictReader(open(P("c.csv"), encoding="utf-8")))
    for r in crows:
        k = key[r["blind_id"]]
        r["cardiologist_decision"] = "value"
        r["cardiologist_lvef"] = str(plan[k["study_id"]]["value"] or auto[k["study_id"]]["lvef"] or 55)
    with open(P("c.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=crows[0].keys()); w.writeheader(); w.writerows(crows)
    out = finalize(plan, P("c.csv"), P("k.csv"), min_spot_n=3)
    assert out["spot_check_auto_audit_only"]["n"] >= 1 and out["spot_check_auto_audit_only"]["agreement"] == 1.0
    print("all ok")

if __name__ == "__main__":
    t()
