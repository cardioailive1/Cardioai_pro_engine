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
Answered by Mayo (Adam Resnick and Tommy Stretch, 2026-10-21):
- ECG waveform (`FACT_ECG_WAVEFORM`): one URL-encoded Base64 record per lead; decode each on its own (URL-decode, Base64-decode, signed 16-bit little-endian, x 4.88 = microvolts). There is **no chunk order**: the slide's `STRING_AGG` does not define a sequence, so records are never joined. Eight `LEAD_ID` values (I, II, V1-V6). Use `WAVEFORM_TYPE = 'Rhythm'` for the full 10-second ECG (the median type is one synthesized beat).
- Sample rate: `ECGSAMPLEBASE` in `FACT_ECG_WAVEFORM_RESTING_MEASUREMENT` (250 or 500 Hz).
- Join to `FACT_ECG`: `ECG_WAVEFORM_FPK` does not join to `ECG_FPK`. Use `PATIENT_CLINIC_NUMBER` through `DIM_PATIENT` to get `PATIENT_DK`, then match `ACQUISITION_TS` to `ECG_DTM`. `ENCOUNTER_NUMBER` comes from the matched `FACT_ECG` row.
- One person across record systems: `PATIENT_CLINIC_NUMBER`. For one row per patient in `DIM_PATIENT`, set `PATIENT_MERGED_FLAG = 'N'`. Use it as `patient_id` so the patient-level split cannot put one person on both sides.
- Cause of death is not populated. Mayo recommends **all-cause death** in place of cardiovascular death. The code now defaults to `death_definition="all_cause"` (event type `all_cause_death`); the clinical lead still has to approve the definition.
- Death and follow-up: `FACT_MORTALITY_STATUS` combines the Mayo record with third-party mortality data (Veritas Data Research). A last live interaction is a completed appointment, performed immunization, lab collection or surgical case.
- MI and stroke: use `condition_occurrence` (SNOMED, OMOP) or `FACT_DIAGNOSIS` (ICD). `FACT_DIAGNOSIS.PRIMARY_DIAGNOSIS_FLAG` gives primary or secondary; `ENCOUNTER_NUMBER` links to `FACT_ENCOUNTERS.ENCOUNTER_TYPE`. Billing diagnoses are in `FACT_CLAIMS_ITEM_DIAGNOSIS`, linked to `FACT_CLAIMS` (`CLAIM_CLASS_NAME` is inpatient or outpatient).
- Dates are shifted by the same amount across a patient's whole record.
- Echo images: link `FACT_ECHO_TEST` to the DICOM images by study date and `PATIENT_CLINIC_NUMBER` (Mayo will explore other ways if this is not enough).

Answered by Mayo (Adam Resnick, 2026-10-28), with the OMOP CDM ETL Mapping Pipeline Guide (also in the Resource Hub):
- OMOP `person_id`: `person.person_source_value` is `DIM_PATIENT.PATIENT_CLINIC_NUMBER`. Join OMOP to the FACT tables on that, never on `person_id` alone. (From the 15 Oct 2026 release the ETL uses `PATIENT_CLINIC_NUMBER` because `PATIENT_DK` is deprecated in some MCP tables.)
- Echo LVEF: filter `FACT_ECHO_RESULTS` to LVEF rows using `DIM_ECHO_TEST_RESULT_TYPE.ECHO_TEST_RESULT_NAME` or `.ECHO_TEST_RESULT_DESCRIPTION`; the number is `ECHO_RESULT_VALUE`. Several LVEF-type rows can exist for one study (method by result type); `lvef_result_types` lists them so a person confirms which to use.
- Echo same-day studies: superseded by Adam's 4 Nov answer below (`DIM_ECHO_TEST` is only a lookup of test types, so it cannot pick the exact study).
- Echo DICOM: listed in `FACT_RADIOLOGY_DICOM_INVENTORY` with `MODALITY = 'US'`; find echo in `STUDY_DESCRIPTION` (`radiology_link.echo_dicom_study_sql`).
- SQL: Mayo (Adam Resnick, 28 Oct): "They use BigQuery client to enable SQL queries against your provisioned data." The Workspace uses the **BigQuery client** to run SQL against the provisioned data, so the project SQL is BigQuery Standard SQL (`training/mayo/bq.query_rows`, dataset as `project.dataset`, tables in backticks).
- Work schema: yes, the project can write saved tables (for example `cardioai_work`); described in `My_First_Query.ipynb`, walked through in the data environment session.
- From the OMOP guide: the `death` table has a date only (`cause_concept_id` always 0) and keeps a death only if its date + 60 days is after the person's last clinical event, so all-cause death is the only workable definition. **Superseded 9 Nov 2026:** in the Workspace the OMOP `death` table holds 25,808 deaths against 253,055 consensus deaths in `FACT_MORTALITY_STATUS` (90% missing; dates agree within 30 days where both exist), so labels use `CONSENSUS_DEATH_DATE` with `CONSENSUS_STATUS = 'deceased'`, exclude `contradicted` patients, and stop at `FOD_DATA_CUTOFF_DATE` (27 Mar 2026). Rules and tests: `training/mayo/mace_label_rules.py`. An unknown birth year is stored as **1800** (month and day NULL): the code treats it as unknown, never as an age. The `measurement` table can hold two rows per lab result (one with a real concept, one with concept 0): match on the concept id and de-duplicate before counting. Diagnoses land in the CDM table that matches the standard concept's domain, not the source table.

Answered by Mayo's ECG Data slides (3 Nov 2026):
- `FACT_ECG` carries `PATIENT_DK`, and `DIM_PATIENT` holds both `PATIENT_CLINIC_NUMBER` and `PATIENT_DK`, so the waveform-to-ECG link in the join bullet above is confirmed: waveform `PATIENT_CLINIC_NUMBER` -> `DIM_PATIENT` -> `FACT_ECG.PATIENT_DK`, then `ACQUISITION_TS` = `ECG_DTM`. `ENCOUNTER_NUMBER` is available only after reaching `FACT_ECG`.
- Use the physiological time: `ACQUISITION_TS` (bedside) and `FACT_ECG.ECG_DTM`. `DATA_AVAILABLE_DTM` and `ECG_DOCUMENT_UPDATE_DTM` are later filing times; do not use them for time-sensitive work such as outcome windows.
- One ECG has several `FACT_ECG` rows: interpretation snippets are stored separately. Order `ECG_INTERPRET_TEXT` by `ECG_INTERPRET_SEQ_NBR`; study-level measurements repeat across those rows and are not separate ECG studies, so collapse to one row per ECG before counting.
- Interpretation: the machine proposes codes and text, and the interpreting **cardiologist** (not the fellow) confirms, edits and signs in MUSE; validated results then flow to the platform. `ECG_INTERPRET_FULL_TEXT` is the combined, signed narrative. `ECG_INTERPRET_CODE_DK` / `DIM_ECG_INTERPRET_CODES` are machine-proposed codes confirmed or edited by the cardiologist. `ECG_INTERPRET_EXTENSION_TEXT` is cardiologist free text only. Searching codes alone misses clinician-added text.
- Resting diagnostic ECGs only (not Holter or remote monitoring): `EKG_TYPE = 'resting'`, `WAVEFORM_TYPE = 'rhythm'`. 250 Hz = 2,500 samples per lead and 500 Hz = 5,000. Decode order: URL-decode, normalise Base64 (`%0A` is a line break), Base64-decode, read as `<i2`, multiply by 4.88. The four derived limb leads (III = II - I, aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2) match `inference.ecg_leads.derive_twelve_lead`.
- Heart rate exists in several columns (clinician-validated, cart-calculated, or computed from the signal); choose one on purpose and record which.

Answered by Mayo (Tommy Stretch, 2026-11-05):
- Rhythm records: usually one record is the complete waveform for a lead. Occasionally a lead has several records for one acquisition. If they are partial, aggregation may be appropriate; if each holds the expected 2,500 or 5,000 samples, treat each as a complete waveform and do not combine. The loader collapses identical complete copies, refuses complete records that differ, and refuses partial records (there is still no chunk order).
- `FACT_ECG_WAVEFORM_RESTING_MEASUREMENT` (`ECGSAMPLEBASE`) joins `FACT_ECG_WAVEFORM` on `PATIENT_DK` and `ACQUISITION_TS` (`ecg_fact_link.ecg_waveform_sql` does this and returns `ECGSAMPLEBASE`).
- Every ECG interpretation on the platform was reviewed and signed by a cardiologist in MUSE before it came over; `ECG_INTERPRET_FULL_TEXT` can be the same report as in MUSE. No signed or unsigned flag is needed.
- Heart rate: use `FACT_ECG.HEART_RATE`, the final clinician-validated value (`ecg_fact_link` returns it as `heart_rate_bpm`).

Answered by Mayo (Adam Resnick, 2026-11-04):
- `PATIENT_DK` for the de-identification CSV comes from `DIM_PATIENT` through `PATIENT_CLINIC_NUMBER` (`PATIENT_MERGED_FLAG = 'N'`). Confirmed.
- Which echo `STUDY_DESCRIPTION` or procedure values count as a transthoracic echo is **our decision**. Use the Term Finder tool in the My First Query notebook (`/data/intro/Introduction_Notebooks/Accelerate`) to find the right ones for our criteria, then have the clinical lead approve the list.
- Same-day echo matching: **correction to the 28 Oct note.** `DIM_ECHO_TEST` is a lookup table for the test type; it holds no information for a specific patient or exam, so `ECHO_PERFORMED_TEST_DK` -> `DIM_ECHO_TEST` cannot identify the exact DICOM study. Do not rely on it. Mayo will discuss this further (4 Nov afternoon).
- Which LVEF result types to prefer is up to our team (`lvef_result_types` lists them).
- The project and dataset names are in the My First Query notebook outline, and Mayo will also send them by email.
- OMOP concepts: the `CONCEPT` table is in BigQuery and many codes map to standard vocabularies (LOINC, SNOMED and others). See the Concept Reference tab of the OMOP CDM ETL Mapping Pipeline Guide. Mayo will run an ATLAS training (it has concept search); https://omophub.com/tools/concept-lookup is an online reference.

Still open:
- [ ] How to match the exact same-day echo study to its DICOM images (Adam: to be discussed with Mayo)
- [ ] Pick the echo `STUDY_DESCRIPTION` values with Term Finder and get the clinical lead to approve them
- [ ] Pick which LVEF result types to use (our team)
- [ ] Read the project and dataset names from the My First Query notebook or Mayo's email
- [x] The de-identification request still uses `PATIENT_DK` (Mayo MCP, 2 Nov: CSV of STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL, required before image access). `FACT_ECG` also carries it (ECG slides, 3 Nov)

Still to find out:
- [ ] OMOP concept IDs for `training/omop_adapter.py`: look them up in the `CONCEPT` table, the Concept Reference tab, ATLAS or omophub
- [ ] Check one decoded Rhythm trace by eye against the signed ECG report, (the loader collapses identical complete duplicates and refuses other duplicates)

```python
from training.mayo.ecg_waveform import assemble_mayo_ecg, resample
sig = assemble_mayo_ecg(rows_for_one_acquisition)   # dicts with LEAD_ID, WAVEFORM_TYPE, WAVEFORM -> (8, n) microvolts
sig = resample(sig, sample_rate_hz, 500)            # 250 Hz ECGs are brought to 500 Hz
```

## 3. Select studies and link reports to images
```python
from training.mayo.radiology_link import cardiac_mri_study_sql, link_reports_to_images
from training.mayo.bq import query_rows                  # Mayo (28 Oct): the Workspace runs SQL through the BigQuery client
rows = query_rows(cardiac_mri_study_sql("<project>.<dataset>"))   # BigQuery SQL; (confirm) the project.dataset in the Workspace; refine the LIKE terms against real PROCEDURE_CODE_TEXT values
import pandas as pd; df = pd.DataFrame(rows)
reports   = df[["DICOM_ACCESSION_NUMBER", "RADIOLOGY_REPORT", "RADIOLOGY_IMPRESSION"]].rename(columns={"DICOM_ACCESSION_NUMBER": "ACCESSION_NBR"}).drop_duplicates().to_dict("records")
inventory = df.to_dict("records")
linked = link_reports_to_images(reports, inventory)
print(len(linked["linked"]), "linked;", len(linked["no_images"]), "no images;", len(linked["not_submittable"]), "not submittable")
```

## 4. Request de-identified images (Mayo TSM)
Order: build the cohort and the pool first (step 3: select the studies, link reports to images, count what is submittable), then choose the sample from the pool, and only then send this request. Request the selected studies only, not the whole pool.
```python
from training.mayo.radiology_link import write_deid_request_csv
# Mayo requires a CSV with STUDY_INSTANCE_UID, PATIENT_DK, MODALITY and STMO_NAME_IN_ADL for each selected study, sent to TSM
# BEFORE image access is granted. The step 3 query returns all four (PATIENT_DK through DIM_PATIENT), so pass the selected rows.
selected_rows = rows   # filter to the studies you chose
print(write_deid_request_csv(selected_rows, "deid_request.csv"))   # written count + rejected rows with reasons
```
Where PATIENT_DK comes from: Mayo's Radiology slides (2 Nov) list `PATIENT_CLINIC_NUMBER` on the DICOM inventory, the header storage and `FACT_IMAGINGSTUDY`, not `PATIENT_DK`. The step 3 query therefore joins `DIM_PATIENT` on `PATIENT_CLINIC_NUMBER` (with `PATIENT_MERGED_FLAG = 'N'` for one row per patient) to get it. **(confirmed by Mayo, 4 Nov)**
Send `deid_request.csv` to TSM. The file must go before image access is granted; the writer refuses rows with a missing value or an image status other than Ready or Available, and removes duplicate studies. Images arrive later; steps 5-9 work on report text meanwhile **(confirm reports are readable now)**.

## 5. Automatic annotation
```python
from training.mayo.label_audit import auto_label_all
records = [{"study_id": r["ACCESSION_NBR"], "kind": "mri", "text": r["RADIOLOGY_REPORT"], "impression": r["RADIOLOGY_IMPRESSION"]} for r in reports]
auto = auto_label_all(records)      # LVEF + findings; addenda, ranges, prior-vs-current EF are flagged
```

## 6. Automatic audit (required)
```python
from training.mayo.label_auto_audit import auto_audit, summarize
res = auto_audit(auto); print(summarize(res))       # pass / warn / fail counts
```
This step cannot be skipped: `export_ds_audit`, `route`, the cardiologist queue, `finalize` and `build_lvef_examples` raise `AutoAuditRequired` if any study lacks an automatic-audit result or the labels did not come from `finalize`.

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


## Linking waveforms to the clinical ECG (`training/mayo/ecg_fact_link.py`)
```python
from training.mayo.bq import query_rows
from training.mayo.ecg_fact_link import (ecg_waveform_sql, fact_ecg_sql, group_waveform_acquisitions,
                                         collapse_fact_ecg, link_waveform_to_fact_ecg)
acq   = group_waveform_acquisitions(query_rows(ecg_waveform_sql("<project>.<dataset>")))   # resting rhythm rows, PATIENT_DK via DIM_PATIENT
facts = collapse_fact_ecg(query_rows(fact_ecg_sql("<project>.<dataset>")))                  # several interpretation rows -> one ECG
r = link_waveform_to_fact_ecg(acq, facts)       # ACQUISITION_TS = ECG_DTM; unmatched and ambiguous are listed, never guessed
print(len(r["linked"]), "linked;", len(r["unmatched"]), "unmatched;", len(r["ambiguous"]), "ambiguous")
```
Each linked entry has `index_time` (the ECG time, never a filing time), the waveform `records` for `assemble_mayo_ecg`, and the collapsed clinical `ecg` (encounter, ordered interpretation text, codes). **(confirm the column names in the Workspace)**

## ECG -> MACE network

0. No image request is needed for this model. The ECG waveforms and labels are tables in the Workspace; the TSM de-identified-image request (step 4) is only for radiology / echo images (the LVEF image model).
1. Build `ecg_rows.csv` in the Workspace, one row per ECG: `patient_id` (PATIENT_CLINIC_NUMBER), `leads_json` (a JSON object `{"I": <WAVEFORM>, "II": ..., "V6": ...}` for the Rhythm recording, values exactly as stored), `sample_rate_hz`, `label_window`, `age`, `sex_male`, `heart_rate_bpm`, `qt_interval_ms`. Take `sample_rate_hz` from the waveform length (`resolve_sample_rate`: 5000 samples = 500 Hz, 2500 = 250 Hz), NOT from `ECGSAMPLEBASE`: in a 484-ECG check, 57 labelled 500 Hz were really 250 Hz. `label_window` comes from `mace_label_rules.mace_label_sql` (positive = first MI/stroke/death 30-89 days after the ECG; imminent, prior-event and no-follow-up ECGs are excluded), then the audit below. Death is `FACT_MORTALITY_STATUS` (OMOP `death` holds only about 10% of deaths). Skip any ECG that has more than one Rhythm record for a lead, and keep a count of them. Split by patient and cap ECGs per patient.
2. Train both variants (12-lead and the single-lead wearable proxy). `--sample-rate` is the common rate every ECG is resampled to:
   `python -m training.train_ecg_mace_mayo --rows ecg_rows.csv --mayo-leads --sample-rate 500 --out weights`
3. Read the printed network-vs-baseline comparison and `weights/cutoff_proposals_*.json`. Only these aggregates leave the Workspace.
4. Clinical team picks cutoffs from the proposal table: `ecg_models.checkpoint.set_tier_cutoffs(path, moderate, high, set_by, note)`.
5. If the network beat the baseline: `approve_checkpoint(path, approver, note)`. Approval is refused for synthetic data, no baseline win, or unset cutoffs. Only then does the live engine load it.


## ECG + echo: second-stage MACE model (LVEF as an input)

Heart failure changes MACE risk, so the most recent EARLIER echo LVEF is added to the ECG network's score. Mayo: more than 100,000 patients have both; link on `PATIENT_CLINIC_NUMBER` and date (dates are shifted equally within a patient); LVEF is in `FACT_ECHO_RESULTS`.

1. Pull echo rows: `FACT_ECHO_RESULTS` joined to `DIM_ECHO_TEST_RESULT_TYPE` (result type name and description) and to `FACT_ECHO_TEST` for the date, as `patient_id` (PATIENT_CLINIC_NUMBER), `echo_date`, `ECHO_RESULT_VALUE`, `ECHO_TEST_RESULT_NAME`, `ECHO_TEST_RESULT_DESCRIPTION` (confirm the join column names in the Workspace). Run `lvef_result_types(rows, type_columns=[...])` and have the clinical lead confirm which types are really LVEF. If a study has several values, the study value is their median (`multi_value="mean"` is the alternative).
2. `studies, st = echo_study_lvef(echo_rows, lvef_column="ECHO_RESULT_VALUE", type_columns=["ECHO_TEST_RESULT_NAME", "ECHO_TEST_RESULT_DESCRIPTION"], study_column=None)` then `rows, ls = link_ecg_to_echo(ecg_rows, studies, max_days_before=365)` (from `training.mayo.ecg_echo_link`). Add `ecg_date` to each ECG row first. Only an echo ON OR BEFORE the ECG is used; an echo after it would leak the outcome. ECGs with no echo are kept (`has_echo` false). Read `ls["share_with_echo"]` and `ls["echo_only_after_the_ecg_not_used"]`.
3. Build examples as usual (`build_examples` carries `echo` through), train the ECG network, then
   `r = fit_stage2(val, val_ecg_scores, test, test_ecg_scores)` (from `training.ecg_echo_stage2`), where the scores are the network's calibrated probabilities.
4. Read `r["vs_ecg_only_all_test_patients"]` and `r["vs_ecg_only_patients_with_echo"]`. Keep the second stage only if `r["beats_ecg_only"]` (whole 95% interval above zero). Only aggregates leave the Workspace.
5. Export and approve: `export_stage2(r, "weights/ecg_echo_stage2_v1.json", ecg_checkpoint_path="weights/ecg_mace_v1.pt", trained_on="Mayo Clinic Platform (real ECG + echo + OMOP outcomes)")`, then the clinical team picks cutoffs (`ecg_models.stage2_checkpoint.set_stage2_cutoffs`) and approves (`approve_stage2`). Approval is refused for synthetic data, no paired win over the ECG-only score, or unset cutoffs. The file records the SHA-256 of the ECG network it was fitted on and is ignored if the live ECG checkpoint is a different file.
6. Live: `POST /api/mace/score` and the clinician dashboard accept the most recent echo on or before the ECG (`echo_lvef`, `days_since_echo`) plus `age` and `sex_male`. With an approved, matching second stage the engine returns the ECG + echo score and the ECG-only score; an echo that is too old (over the training window) is not used, and the patient is scored as one with no echo. Without age and sex, or without an approved second stage, the ECG-only model scores. A negative `days_since_echo` is rejected (an echo after the ECG would leak the outcome).

## Reviewing the MACE outcome labels (same two-tier design as echo)

Module: `training/mayo/mace_label_audit.py`. Run inside the Workspace; the files hold patient-level evidence.

0. Sample and records: `audit_sample_sql` / `build_audit_records` (`mace_audit_sample.py`) draw a stratified one-ECG-per-patient sample (about 700) and build the records and evidence. This is a SAMPLE audit: it measures how often the rule agrees with reviewers; systematic errors are fixed in the rule, not by hand.
0b. Corroboration (`mace_corroboration.py`): `corroboration_sql` + `tier_records` tier each positive as corroborated (elevated troponin or PCI/CABG for MI, brain imaging for stroke, consensus source for death), conflict, or code_only. Send only conflict, code_only and flagged rows to the data scientists; the cardiologist sample stays.
1. Records come from step 0 (they carry `index_date`).
2. `items = build_review_items(records, condition_rows, death_rows, mapping, visit_rows)` collects the OMOP evidence per record.
3. `aa = auto_audit(items)` (required; the later steps refuse to run without it): automatic checks (death with no usable cause, event after death, window within a week of 30/90 days, single supporting code, non-inpatient setting, MI+stroke together, prior event at the washout edge, code present but labeled no event).
4. Tier 1: `export_ds_audit(items, "ds.csv", aa)`. Each data scientist fills `ds_decision` (agree / disagree / uncertain / exclude); for disagree add `ds_event_type` (none, myocardial_infarction, stroke, all_cause_death, or cardiovascular_death if the batch uses that definition) and `ds_days_to_event`. Optional lighter pass: `triage_ds_scope` (always keeps events and non-pass records).
5. `plan = route(items, read_ds_audits(["ds.csv", ...]), aa)`.
6. Tier 2: `export_cardiologist_queue(items, plan, "card.csv", "key.csv")`. Blinded, shuffled; spot-check drawn separately from agreed events and agreed non-events. The cardiologist fills `cardiologist_decision` (event / none / exclude), plus type and days for an event. Keep `key.csv` with the data scientists.
7. `result = finalize(plan, "card.csv", "key.csv")`; passes at >= 95% spot-check agreement over >= 20 checks with nothing unresolved.
8. `fixed = apply_to_records(records, result)` -> `LabelValidator` -> training.


## Running trained models in the dashboards (after training and approval)

What exists: the clinician dashboard (Cardiologist / Physician page) has "Run MACE Prediction on an ECG", the payer portal has
"Score Members with the Claims-Risk Model" (Population Risk page) and "Controlled Estimate" (Cost Avoidance page). Each shows
`GET /api/models/status` so nobody mistakes a placeholder for a trained model. Until a model is live, the ECG panel scores
nothing from the waveform and says so; the member panel labels every score a placeholder.

To make a trained model live in the web service, all four must be true (the status strip shows which one is missing):
1. **Trained and approved in the Workspace**: checkpoint plus its `.json` sidecar with `approved_for_live: true` (cutoffs set by the clinical team, gates passed).
2. **Cleared to leave the Workspace.** Only code and aggregates may leave by default. Model weights learned from Mayo data are neither, so ask Mayo (Adam) in writing whether and how the approved weights may be exported before copying anything. Do not move them until that is answered.
3. **Files in the weights folder** of the web service (`CARDIOAI_WEIGHTS_DIR`, default `weights/`): `ecg_mace_v1.pt`, `ecg_mace_1lead_v1.pt`, `echo_lvef_v1.pt`, `claims_risk_v1.joblib`, each with its `.json`. On Render this needs a persistent disk or a rebuild with the files baked in.
4. **Model runtime installed**: set `WITH_MODEL_RUNTIME=1` (build arg / Render env var) so `requirements-inference.txt` (torch, scikit-learn) is installed. Without it the status strip says `runtime missing` and the placeholder keeps scoring.

ECG input to the panel: a .csv with one column per lead named `I, II, V1..V6` (Mayo's 8 stored leads; III, aVR, aVL and aVF are
calculated), or all 12, or lead I alone; .json `{"I": [...], ...}` also works. Choose 250 or 500 Hz to match the study. The
model resamples to its training rate.

The member roster for the payer panel is a .csv: `member_id, age, sex, diagnosis_codes (ICD-10, semicolon separated), inpatient_visits_12m, ed_visits_12m`.
The controlled estimate takes `member_id, treated, pre_pmpm, post_pmpm` plus any numeric pre-program covariates (age, risk_score...).
