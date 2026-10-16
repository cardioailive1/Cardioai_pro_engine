# Mayo Clinic Platform runbook - run in this order

Run everything inside the Mayo Workspace (JupyterLab). Only code goes in and only code/aggregate results come out.
Items marked **(confirm)** are assumptions to settle with Mayo (Data Environment / Radiology training). Never commit
CSVs, report text, IDs or images. Run all commands from the repo's `backend/` folder.

## 0. Before first sign-in (not in the Workspace)
- [ ] DUA form submitted for every user, including all 5 annotators (cardioailive.com emails)
- [ ] Attended: Radiology, Data Environment, ECG and Lab Test trainings
- [ ] If a container is needed: build it self-contained and save as a tar (`docker save -o cardioai.tar IMAGE`); `docker pull` does not work in the Workspace

## 1. Start the environment
1. Open the Workspace, start JupyterLab. Turn on the GPU when you reach step 11 (A100 40 GB attaches automatically if free).
2. Put the repo code in the Workspace (code only), then in a terminal:
```bash
cd cardioai_pro_engine/backend
pip install -r requirements.txt -r requirements-training.txt -r requirements-annotation.txt
python -m training.mayo.test_report_labels && python -m training.mayo.test_label_audit && python -m training.mayo.test_label_auto_audit
python -m training.mayo.test_lvef_dataset      # needs torch + pydicom
```
All four must print `all ok`. 3D Slicer and ITK-SNAP are pre-installed (desktop session).

## 2. Look up what is still unconfirmed
- [ ] Which table holds `PATIENT_DK` (try `FACT_IMAGINGSTUDY`)
- [ ] How `FACT_ECHO_TEST` links to echo images
- [ ] ECG Base64 layout: take 3-5 rows and run
```python
from training.mayo.ecg_waveform import probe_waveform, decode_waveform
print(probe_waveform(b64_string))     # candidate dtype / lead count / byte order; confirm with ECG training, check a trace by eye
x = decode_waveform(b64_string, dtype="int16", n_leads=12, layout="lead_major")   # use the confirmed layout
```
- [ ] OMOP concept IDs for `training/omop_adapter.py` (CONCEPT table / Schema Visualizer)

## 3. Select studies and link reports to images
```python
from training.mayo.radiology_link import cardiac_mri_study_sql, link_reports_to_images
df = spark.sql(cardiac_mri_study_sql("<schema>")).toPandas()     # (confirm) Spark session name; refine the LIKE terms against real PROCEDURE_CODE_TEXT values
reports   = df[["DICOM_ACCESSION_NUMBER", "RADIOLOGY_REPORT", "RADIOLOGY_IMPRESSION"]].rename(columns={"DICOM_ACCESSION_NUMBER": "ACCESSION_NBR"}).drop_duplicates().to_dict("records")
inventory = df.to_dict("records")
linked = link_reports_to_images(reports, inventory)
print(len(linked["linked"]), "linked;", len(linked["no_images"]), "no images;", len(linked["not_submittable"]), "not submittable")
```

## 4. Request de-identified images (Mayo TSM)
```python
from training.mayo.radiology_link import write_deid_request_csv
# rows need STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL (join PATIENT_DK from the table found in step 2)
print(write_deid_request_csv(selected_rows, "deid_request.csv"))   # written count + rejected rows with reasons
```
Send `deid_request.csv` to TSM. Images arrive later; steps 5-9 work on report text meanwhile **(confirm reports are readable now)**.

## 5. Automatic annotation
```python
from training.mayo.label_audit import auto_label_all
records = [{"study_id": r["ACCESSION_NBR"], "kind": "mri", "text": r["RADIOLOGY_REPORT"], "impression": r["RADIOLOGY_IMPRESSION"]} for r in reports]
auto = auto_label_all(records)      # LVEF + findings; addenda, ranges, prior-vs-current EF are flagged
```

## 6. Automatic audit (optional)
```python
from training.mayo.label_auto_audit import auto_audit, summarize
res = auto_audit(auto); print(summarize(res))       # pass / warn / fail counts
```

## 7. Tier 1 - data scientists audit
```python
from training.mayo.label_audit import export_ds_audit, read_ds_audits, route, audit_summary
export_ds_audit(auto, "ds_audit.csv", auto_audit=res)       # everything, riskiest first (default). Give each auditor a copy, fill `auditor`
# OPT-IN lighter tier 1 (only after spot-check data shows the auto-audit is reliable):
#   from training.mayo.label_audit import triage_ds_scope
#   scope = triage_ds_scope(auto, res); export_ds_audit(auto, "ds_audit.csv", auto_audit=res, ds_scope=scope)
```
Auditors set `ds_decision` = agree / disagree (+ `ds_lvef`) / uncertain / exclude, then:
```python
audits = read_ds_audits(["ds_audit_a.csv", "ds_audit_b.csv"])
print(audit_summary(auto, audits))                          # coverage must be 1.0 (or use ds_scope)
plan = route(auto, audits, auto_audit=res)                  # add ds_scope=scope only if you used triage
```

## 8. Tier 2 - cardiologist
```python
from training.mayo.label_audit import export_cardiologist_queue, finalize
print(export_cardiologist_queue(auto, plan, "cardiologist.csv", "cardiologist_key.csv"))   # blinded + shuffled; refuses if any study is unaudited
```
Give the cardiologist **only** `cardiologist.csv`; keep `cardiologist_key.csv` with the data scientists. They fill `cardiologist_decision` (`value` + `cardiologist_lvef`, or `exclude`), then:
```python
out = finalize(plan, "cardiologist.csv", "cardiologist_key.csv")
print(out["spot_check"], out["passed"], len(out["unresolved"]))   # passes only at >=95% spot-check agreement, >=20 checks, nothing unresolved
```
If `passed` is False: widen the audit or fix the labeler rules, re-run steps 5-8. Do not train on a failed batch.

## 9. Image outlines (only if segmentations are needed)
No segmentations exist at Mayo. Annotators (5 seats, outside annotators) use 3D Slicer or the OHIF/`monailabel` flow on de-identified images. Start `monailabel` from the Workspace terminal **(confirm exact command/ports in Data Environment training)**.

## 10. Build training data (once de-identified images arrive)
```python
from training.mayo.lvef_dataset import build_lvef_examples, DicomCineLVEFDataset
examples = build_lvef_examples(linked["linked"], out["labels"])        # only reviewed labels
ds = DicomCineLVEFDataset(examples, path_for_sop=lambda sop: f"<dicom_dir>/{sop}.dcm")   # (confirm) where delivered files land
```

## 11. Train (GPU on)
```python
from training.train_mobile_echo import train_mobile_echo_net
res = train_mobile_echo_net(n_epochs=30, batch_size=16, dataset=ds, n_val=max(2, len(ds) // 10))
print(res["correlation"])
```
Plumbing only until there are hundreds of reviewed studies; do not quote results from a small run.

## 12. Leaving the Workspace
Only code, aggregate metrics (spot-check agreement, correlation) and approved model exports leave, via Mayo's export process. No report text, IDs, CSVs or images.


## ECG -> MACE network (after Mayo confirms the waveform layout)

1. Build `ecg_rows.csv` in the Workspace: `patient_id, waveform_b64, label_window, age, sex_male, heart_rate_bpm, qt_interval_ms` (label_window from OMOP outcomes via `label_schema`; the last two columns feed the baseline). Confirm the layout first with `ecg_waveform.probe_waveform` on a few rows.
2. Train both variants (12-lead and the single-lead wearable proxy):
   `python -m training.train_ecg_mace_mayo --rows ecg_rows.csv --dtype <..> --n-leads 12 --layout <..> --sample-rate <Hz from Mayo> --out weights`
3. Read the printed network-vs-baseline comparison and `weights/cutoff_proposals_*.json`. Only these aggregates leave the Workspace.
4. Clinical team picks cutoffs from the proposal table: `ecg_models.checkpoint.set_tier_cutoffs(path, moderate, high, set_by, note)`.
5. If the network beat the baseline: `approve_checkpoint(path, approver, note)`. Approval is refused for synthetic data, no baseline win, or unset cutoffs. Only then does the live engine load it.
