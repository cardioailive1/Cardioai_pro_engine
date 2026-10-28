# Mayo Clinic Platform Workspace - training and annotation setup

Source: email from Adam Resnick (Mayo Clinic Platform), 2026-10-14.

## Confirmed by Mayo
| Item | Answer |
|---|---|
| Desktop session, 3D Slicer, ITK-SNAP | Available, already installed |
| pip install `monailabel`, `highdicom` | Allowed (see `requirements-annotation.txt`) |
| Local web apps via JupyterLab (OHIF, CVAT) | Allowed on localhost |
| Docker / podman | Allowed, but `docker pull` does not work: images must be self-contained |
| Annotator seats | 5 additional, each needs a cardioailive.com email |
| Mayo annotators | Not possible in our environment; hire outside annotators |
| GPU | Shared-reservation A100 40 GB; other machine types on request |
| Existing annotations | None. Labels come from reports/notes (below) |
| Public datasets (EMIDEC, MyoPS, ACDC) | Keep outside the environment; request import only if essential, subject to licences |

## Second reply from Mayo (2026-10-15)
| Question | Answer |
|---|---|
| Where are cardiac MRI / echo DICOM images? | Not in a table. Inventoried in `FACT_RADIOLOGY_DICOM_INVENTORY`; files are requested from that inventory. Covered in the Radiology training session and Resource Hub slides |
| Which column links a study to its report? | Radiology slides: `FACT_RADIOLOGY.ACCESSION_NBR` = `FACT_RADIOLOGY_DICOM_INVENTORY.DICOM_ACCESSION_NUMBER` (see `radiology_link.py`). FACT_ECHO_TEST link still unconfirmed |
| Docker as a saved tar | Same import process as other self-contained images (yes) |
| Annotator onboarding | Same as every data user, including the DUA form submission (Mayo smartsheet form) for each of the 5 accounts |
| A100 booking / limits | Turn the GPU on; it attaches automatically if a machine is free. No usage limit stated; covered in Data Environment training |

## Where the data is
- ECG waveforms: Base64-encoded string, must be decoded -> `training/mayo/ecg_waveform.py`
  (binary layout still to be confirmed from the ECG training / Mayo)
- Radiology reports (cardiac MRI): `FACT_RADIOLOGY`
- Echo notes: `FACT_ECHO_TEST.ECHO_NOTE`
- Labs: look up by Reporting name at mayocliniclabs.com (Clinical & Interpretive, Fees & Codes tabs)
- Training material: ECG and Lab Test trainings in the Mayo Resource Hub (Smartsheet); Radiology slides

## What this repo now does about it
- Automatic annotation = `training/mayo/report_labels.py`: pulls LVEF (with HF category), and
  MRI/echo findings (LGE, infarct/scar, edema, effusion, LVH, valve disease, wall motion) from report
  text, negation-aware, with the matched text as provenance. Ambiguous cases are flagged
  `needs_review` for the human annotators. It is a pre-labeler, not ground truth.
- Annotator review = `training/mayo/review_queue.py`: exports every flagged item plus a 10% random QA sample
  of auto-accepted items to a CSV (decision: accept / correct / exclude, corrected_lvef). Annotators open it in
  the Workspace JupyterLab; `apply_reviews` merges decisions and reports the pre-labeler's measured agreement.
  Nothing needs installing for this or for the text auto-annotation (Python standard library + numpy only).
  `monailabel`/`highdicom` are needed only for image outlines.
- Image linking = `training/mayo/radiology_link.py` (from the Radiology slides): joins reports to the DICOM
  inventory on accession number, keeps only images with IMAGE_DEID_STATUS Ready/Available, and gives a starter
  BigQuery SQL for cardiac MRI and echo (run through `training/mayo/bq.query_rows`; Mayo's Workspace uses the BigQuery client). Reports with an addendum/edit are always sent to review. Radiology reports are
  also in OMOP `note.note_text`; the DICOM inventory is not in OMOP.
- Training hookup = `training/mayo/lvef_dataset.py`: reviewed LVEF labels + linked DICOM -> a torch dataset that
  `train_mobile_echo_net(dataset=...)` accepts (default synthetic behaviour unchanged). Only human-reviewed
  labels are used. Series choice and cine ordering for MRI stacks are simple defaults to refine on real data.
  Test: `python -m training.mayo.test_lvef_dataset` (needs torch + pydicom).
- De-identification request = `write_deid_request_csv` in `radiology_link.py`: writes the CSV Mayo (TSM) asked for -
  STUDY_INSTANCE_UID, PATIENT_DK, MODALITY, STMO_NAME_IN_ADL - one row per selected study, rejecting rows with
  missing values or a status other than Ready/Available. The real rows come from the Workspace.
- Two-tier label audit = `training/mayo/label_audit.py`. Tier 1: data scientists audit EVERY auto label (agree /
  disagree + their LVEF / uncertain / exclude); the cardiologist file cannot be built until all studies are covered.
  Tier 2: the cardiologist adjudicates only disagreements (auditor vs auto, or auditor vs auditor), uncertain cases
  (auditor unsure or labeler-flagged) and a small blinded spot-check (default 5%, minimum 5) of agreed studies.
  Their CSV is shuffled with random ids and shows no auto label, auditor decision or queue reason; a separate key
  file stays with the data scientists. `finalize` gives final labels (cardiologist overrides consensus), exclusions,
  unresolved studies and the spot-check agreement (with 95% interval); the batch passes only at >=95% agreement over
  >=20 spot-checks with nothing unresolved. Feed `finalize(...)['labels']` to `lvef_dataset.build_lvef_examples`.
  This replaces the single-tier `review_queue` for anything used in training. Test: `python -m training.mayo.test_label_audit`.
- Automatic audit (required after automatic labeling) = `training/mayo/label_auto_audit.py`, run before the data scientists. Every later step refuses to run without a result for every study (`audit_gate.py`). Independent checks per
  label: value present, plausible range, prior/comparison wording near the EF, multiple EF values, qualitative wording
  vs number ("severely reduced" vs 58%), impression vs body, structured EF measurement, identical text labeled
  differently. Default: it only ADDS columns to the data-scientist CSV (status, notes, suggested LVEF) and lists risky
  studies first - they still audit everything. Opt-in (`triage_ds_scope`): they audit every non-pass study plus a random
  20% of passes; the rest count as audited by "auto-audit", are oversampled (10%) in the blinded cardiologist
  spot-check, and agreement for them is reported separately. An automatic fail always goes to the cardiologist, even
  if a data scientist agreed. Qualitative LVEF ranges are heuristics for the clinical team to confirm.
  Test: `python -m training.mayo.test_label_auto_audit`.
- Step-by-step commands: `training/mayo/RUNBOOK.md`.
- Tests: `python -m training.mayo.test_report_labels` (from `backend/`).

## Still open (not done here)
0. Echo-note-to-image link (FACT_ECHO_TEST), the DICOM request process, and DUA forms for the 5 annotators.
1. Confirm the ECG Base64 layout (dtype, lead count, interleaving, sampling rate) on real rows.
2. Image outlines: no segmentations exist, so cardiac MRI/echo outlines need monailabel + outside annotators
   (3D Slicer/OHIF); GPU pre-drawing runs on the A100.
3. Fill the OMOP concept_ids in `training/omop_adapter.py` from the Workspace CONCEPT table.
4. Keep PHI out of this repo: report text stays in the Workspace; only code leaves.
