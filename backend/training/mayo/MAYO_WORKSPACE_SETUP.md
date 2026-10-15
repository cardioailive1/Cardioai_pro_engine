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
  Spark SQL for cardiac MRI. Reports with an addendum/edit are always sent to review. Radiology reports are
  also in OMOP `note.note_text`; the DICOM inventory is not in OMOP.
- Tests: `python -m training.mayo.test_report_labels` (from `backend/`).

## Still open (not done here)
0. Echo-note-to-image link (FACT_ECHO_TEST), the DICOM request process, and DUA forms for the 5 annotators.
1. Confirm the ECG Base64 layout (dtype, lead count, interleaving, sampling rate) on real rows.
2. Image outlines: no segmentations exist, so cardiac MRI/echo outlines need monailabel + outside annotators
   (3D Slicer/OHIF); GPU pre-drawing runs on the A100.
3. Fill the OMOP concept_ids in `training/omop_adapter.py` from the Workspace CONCEPT table.
4. Keep PHI out of this repo: report text stays in the Workspace; only code leaves.
