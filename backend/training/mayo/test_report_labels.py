"""Run: python -m training.mayo.test_report_labels  (from backend/)"""
from training.mayo.report_labels import extract_lvef, annotate_report, ef_category
import numpy as np, base64
from training.mayo.ecg_waveform import decode_waveform, probe_waveform

def t():
    a = extract_lvef("Left ventricular ejection fraction is 35%.")
    assert a.value == 35 and a.category == "reduced_le40" and not a.needs_review
    b = extract_lvef("LVEF estimated at 55-60 %")
    assert b.value == 57.5 and b.category == "preserved_ge50" and not b.needs_review
    c = extract_lvef("Prior EF 30%. Current EF 45%.")
    assert c.needs_review
    d = extract_lvef("EF 48-52%")
    assert d.needs_review
    assert extract_lvef("Ejection fraction not reported.").value is None
    r = annotate_report("There is no late gadolinium enhancement. LVEF 60%.", "mri")
    st = {f["finding"]: f["state"] for f in r["findings"]}
    assert st["late_gadolinium_enhancement"] == "absent" and not r["needs_review"]
    r = annotate_report("Subendocardial LGE in the LAD territory. LVEF 38%.", "mri")
    assert {f["finding"]: f["state"] for f in r["findings"]}["late_gadolinium_enhancement"] == "present"
    assert ef_category(41) == "mildly_reduced_41_49"
    x = (np.arange(12 * 5000) % 200 - 100).astype("<i2").reshape(12, -1)
    s = base64.b64encode(x.tobytes()).decode()
    assert np.array_equal(decode_waveform(s).astype(int), x)
    il = decode_waveform(base64.b64encode(x.T.copy().tobytes()).decode(), layout="interleaved")
    assert np.array_equal(il.astype(int), x)
    assert any(p["dtype"] == "int16" and p["n_leads"] == 12 for p in probe_waveform(s))
    import csv, tempfile, os
    from training.mayo.review_queue import export_review_queue, apply_reviews
    recs = [{"study_id": "A", "kind": "echo", "text": "Prior EF 30%. Current EF 45%."},
            {"study_id": "B", "kind": "echo", "text": "LVEF 60%."},
            {"study_id": "C", "kind": "mri", "text": "Not reported."}]
    p = os.path.join(tempfile.mkdtemp(), "q.csv")
    st = export_review_queue(recs, p, qa_fraction=1.0)
    assert st == {"total": 3, "flagged": 2, "qa_sample": 1}, st
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["decision"] = {"A": "correct", "B": "accept", "C": "exclude"}[r["study_id"]]
        if r["study_id"] == "A": r["corrected_lvef"] = "45"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    out = apply_reviews(p)
    assert out["labels"]["A"]["category"] == "mildly_reduced_41_49" and out["labels"]["B"]["lvef"] == 60
    assert out["excluded"] == ["C"] and out["qa_agreement"] == 1.0 and not out["pending"]
    assert annotate_report("LVEF 55%. ADDENDUM: corrected.", "echo")["needs_review"]
    from training.mayo.radiology_link import link_reports_to_images, cardiac_mri_study_sql
    out = link_reports_to_images(
        [{"ACCESSION_NBR": "1"}, {"ACCESSION_NBR": "2"}, {"ACCESSION_NBR": "3"}],
        [{"DICOM_ACCESSION_NUMBER": "1", "STUDY_INSTANCE_UID": "S", "SERIES_INSTANCE_UID": "x", "SOP_INSTANCE_UID": "y", "IMAGE_DEID_STATUS": "Ready"},
         {"DICOM_ACCESSION_NUMBER": "2", "STUDY_INSTANCE_UID": "T", "IMAGE_DEID_STATUS": "Not Available"}])
    assert list(out["linked"]) == ["1"] and out["not_submittable"] == ["2"] and out["no_images"] == ["3"]
    assert "ACCESSION_NBR = i.DICOM_ACCESSION_NUMBER" in cardiac_mri_study_sql()
    print("all ok")

if __name__ == "__main__":
    t()
