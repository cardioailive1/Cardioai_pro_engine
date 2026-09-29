# CardioAI Pro — Cohort Study Designs for Mayo Clinic Platform Training

Three notebook-scoped cohort studies, each mapped to code already built in
this project (`training/label_schema.py`, `training/omop_adapter.py`,
`imaging_models/`, `fairness/bias_audit.py`). Notebook 1 is ready to design
in full detail today. Notebook 2 has two open questions that must be
resolved with Mayo before its cohort can be finalized — flagged explicitly
below, not glossed over. Notebook 3 depends on Notebook 1 (and eventually
Notebook 2) producing a trained model to validate.

---

## Notebook 1 — ECG Risk Model Cohort Study

### Study design
Retrospective cohort study using de-identified, OMOP CDM-structured EHR
data accessed via Mayo Clinic Platform_Discover. Each **index observation**
(not each patient) is a unit of analysis — a patient with five ECGs over
three years contributes up to five index observations, each independently
assessed for washout and followed forward for an outcome, exactly as
`training/label_schema.py`'s `MACELabelRecord` already models it.

### Source population
Adults (age ≥ 18 at index) with at least one structured heart rate
measurement and one QTc (or QT interval) measurement in the OMOP CDM
`MEASUREMENT` table, drawn from Mayo's available EHR date range. HRV is
explicitly **not** a source-population requirement — see the "expected
data gaps" note below.

### Index date definition
The date of a qualifying `MEASUREMENT` group (see
`training/omop_adapter.py`'s `group_by_visit()` — grouped by
`visit_occurrence_id` when populated, else by `(person_id, measurement_date)`
as a documented fallback).

### Inclusion criteria
| # | Criterion | Rationale |
|---|---|---|
| I1 | Age ≥ 18 at index | Adult MACE risk models; pediatric cardiac risk is a different clinical problem entirely |
| I2 | At least `heart_rate_bpm` and `qt_interval_ms` present at index | The two features present at essentially every institution's structured vitals — see gap note below for HRV |
| I3 | ≥ 90 days of potential follow-up from index to the end of available data | Matches `LabelValidator`'s `MIN_FOLLOWUP_DAYS_FOR_NEGATIVE` — a censored record can't be trusted as a true negative without this |

### Exclusion criteria (all enforced by the existing, unmodified `LabelValidator`)
| # | Criterion | `LabelValidator` mechanism |
|---|---|---|
| E1 | Qualifying MACE event (MI or stroke) within 90 days **before** index | `prior_mace_within_washout` — this observation is mid-event, not a clean "time zero" |
| E2 | Censored record with < 90 days actual follow-up | `insufficient_followup_for_negative` |
| E3 | Feature value outside the physiologic range already enforced live (`ingestion/quality.py`'s `SCHEMAS`) | `feature_out_of_range:<field>` |
| E4 | Missing any of the three subgroup fields (`race_ethnicity`, `gender`, `age_band`) | `missing_subgroups:<fields>` — required so Notebook 3's bias audit can run on this same cohort unchanged |

### Outcome definition
3-point MACE by default — **cardiovascular death, myocardial infarction,
non-fatal stroke** — matching `training/label_schema.py`'s
`MACE_QUALIFYING` default. This is a real point of clinical variation
(4-point adds revascularization, 5-point adds unstable-angina
hospitalization); the clinical team should confirm the composite
definition explicitly rather than inherit the code's default silently.
Ascertained via OMOP `CONDITION_OCCURRENCE` (MI, stroke) and `DEATH`
(cardiovascular cause), exactly as `omop_adapter.py`'s `_find_outcome()`
already implements.

### Labeling windows and the early-detection evaluation set
Every included observation is classified into exactly one window:
`imminent_0_30`, `early_30_90`, `late_90_365`, or `none_observed`.
**Only `early_30_90` and `none_observed` records enter the primary model
evaluation** (`early_detection_eval_set()`) — `imminent_0_30` records are
deliberately excluded from that specific evaluation, because a model that
is simply good at detecting current acute abnormality would trivially
score well there without ever demonstrating genuine early detection. This
is the single most important methodological choice in the whole study;
see that module's docstring for the full argument.

### Predictor variables at index
`heart_rate_bpm`, `qt_interval_ms`, `hrv_sdnn_ms` (when present).

**Expected data gap, stated honestly in advance**: HRV is rarely a
discrete structured OMOP measurement at most institutions — it typically
requires raw ECG waveform analysis rather than manual/automated vitals
documentation. Expect a materially smaller subset of the cohort to have
`hrv_sdnn_ms` populated than `heart_rate_bpm`/`qt_interval_ms`. Design the
analysis to report model performance both with and without HRV, not to
silently drop patients missing it and shrink the cohort without
comment.

### Subgroup dimensions
`race_ethnicity` (`White`, `Black`, `Hispanic`, `Asian`,
`Other/Unspecified`), `gender` (`Female`, `Male`), `age_band` (`<50`,
`50-64`, `65+`) — identical vocabulary to `fairness/bias_audit.py`'s
`SUBGROUP_DIMENSIONS`, by design, so Notebook 3 runs against this cohort
without any translation layer.

### Sample size
Two independent constraints, and the binding one is usually the second:

1. **Events-per-variable (EPV).** A well-established biostatistics rule
   of thumb for regression-type models asks for roughly 10 outcome events
   per predictor variable to avoid overfitting. With 3 predictors
   (HR/QTc/HRV), that's a bare floor of **≥ 30 events in the
   `early_30_90` window** — enough to fit `benchmark_model.py`'s logistic
   regression without it being purely noise-fit, not enough to trust a
   more expressive model or a reliable subgroup breakdown.
2. **Subgroup cell coverage.** `SUBGROUP_DIMENSIONS` has 5 × 2 × 3 = 30
   cells. `LabelValidator` already flags any cell under 10 records as a
   dataset-health issue. For Notebook 3's bias audit to say anything
   trustworthy about, say, sensitivity for Black patients specifically,
   that cell alone needs a real number of `early_30_90` events, not just
   records — likely pushing the true target well into the **hundreds to
   low thousands of `early_30_90` positives**, not the 30-event EPV
   floor. State the achieved N per cell in the notebook explicitly rather
   than proceeding past a validator warning silently.

### Train/validation/test split
Patient-level (`training/train_model.py`'s `patient_level_split()`) —
the same patient's records never span train and test. Stratify by
subgroup where cell sizes allow it.

### Benchmark
`training/benchmark_model.py`'s logistic regression is the statistical
floor `train_model.py`'s candidate model must clear — verified on
synthetic data to matter (candidate beat benchmark on specificity/PPV at
a matched sensitivity, not just raw AUC). It is explicitly **not** a
stand-in for the real external clinical instruments (HEART, TIMI, GRACE),
which need variables (troponin, history, ECG morphology) this cohort
doesn't capture; a genuine external validation against those needs a
separate cohort extension, not this one.

---

## Notebook 2 — Imaging Models (CT / Echo) Cohort Study

### Two open questions that block finalizing this cohort — resolve first
1. **Does Mayo Clinic Platform_Discover expose raw DICOM pixel arrays**,
   not just structured imaging metadata/reports? Everything below assumes
   yes; if the answer is no, this notebook's Stage 1 (below) cannot be
   designed as written and needs a different data-access path confirmed
   with Mayo directly.
2. **Is GPU-backed compute available in the Workspace notebook
   environment?** Self-supervised pretraining (DINO for CT, MAE for echo)
   at real scale needs it; if Workspace is CPU-only, Stage 1 either runs
   elsewhere (under whatever data-use terms permit exporting de-identified
   imaging data that way) or doesn't run as designed.

Everything below is written assuming both resolve favorably — restated
as assumptions, not confirmed facts.

### Why this is a two-stage cohort, unlike Notebook 1
`imaging_models/ct_model.py` and `echo_model.py` follow a self-supervised
pretrain → supervised fine-tune pattern (DINO-LG/CARD-ViT for CT,
Echo-Vision-FM for echo). The two stages need **different cohorts** with
different requirements.

### Stage 1 — self-supervised pretraining cohort (no labels needed)
**Population**: every technically adequate CT (cardiac-gated /
calcium-scoring protocol) or echo study in the de-identified imaging
archive — no diagnostic label required, which is the entire point of
choosing a self-supervised method: unlabeled scans are far easier to
obtain at scale than expert-annotated ones.

**Inclusion**: correct modality/protocol (gated cardiac CT; standard
transthoracic echo views for echo), technically complete DICOM series.

**Exclusion**: corrupted or truncated series, wrong protocol, non-cardiac
studies misfiled under a cardiac order.

**Target size**: real published precedent gives a realistic range, not a
guess — DINO-LG pretrained on 914 real CT scans (700 gated + 214
non-gated); Echo-Vision-FM pretrained at MIMIC-IV-ECHO scale, several
orders of magnitude larger. An institution-scale CT cohort in the
low-thousands is a defensible target for the CT path; echo, given its
lighter per-study weight and typically higher study volume, can likely
support a substantially larger Stage 1 cohort — confirm actual archive
volume with Mayo before committing to a number.

### Stage 2 — supervised fine-tuning cohort (labeled)
**Population**: the subset of Stage 1's cohort that *also* has a
structured ground-truth label at the time of the study.

- **CT**: Agatston/coronary artery calcium score. If not already a
  discrete structured OMOP field, this may require NLP extraction from
  radiology report text — a real, separate methodological workstream to
  scope explicitly, not assume away.
- **Echo**: LVEF (commonly captured as a structured `MEASUREMENT` via a
  LOINC-mapped concept, where the institution codes it) and/or a
  wall-motion-abnormality flag.

**A real design difference from Notebook 1, worth stating explicitly**:
this label is a *contemporaneous finding* (what the image shows, now),
not a *forward-looking outcome* (what happens next). No washout, no
follow-up window, no censoring logic applies here — `label_schema.py`'s
machinery doesn't transfer to this stage; a separate, simpler labeled
dataset structure is needed (image → ground-truth reading, no temporal
window).

**Target size**: DINO-LG's downstream supervised classification and
Echo-Vision-FM's EchoNet-Dynamic fine-tuning precedent (order of
thousands of labeled studies) are the realistic reference points — this
stage needs meaningfully fewer labeled examples than Stage 1 needs
unlabeled ones, since the backbone's representations are already learned
before fine-tuning starts.

### Split strategy
Patient-level, same as Notebook 1 — a patient with both a Stage 1
pretraining appearance and a Stage 2 labeled study must not have that
study split across train and test.

---

## Notebook 3 — Validation (Platform_Validate / Bias-Fairness)

### Population
The **held-out test set** from Notebook 1 (and, once it exists, Notebook
2) — never touched during training or hyperparameter selection. This is
not a new cohort query; it's the reserved slice of the cohorts already
defined above.

### Design
Cross-sectional evaluation of the trained model's performance, stratified
by each `SUBGROUP_DIMENSIONS` cell, using the exact same mechanism
already built and tested in `fairness/bias_audit.py`: subgroup confusion
matrices, sensitivity gap (max − min across subgroups) against a defined
equity target, and — when the gap exceeds that target —
`remediate_thresholds()`'s per-subgroup threshold recalibration, followed
by re-validation to confirm the gap actually closed.

### Minimum cell size for a trustworthy audit
The same constraint as Notebook 1's sample-size section, restated because
it's the actual gating factor for whether Notebook 3 can say anything
meaningful: a subgroup cell with too few `early_30_90` events produces a
sensitivity estimate that's mostly noise. `LabelValidator`'s existing thin-
coverage warning is a floor, not a target — report the achieved N per
cell alongside every subgroup metric in the notebook, and mark any cell
below a pre-agreed threshold (state it explicitly, e.g. 10 or 30 events)
as "insufficient for reliable audit" rather than reporting a
confident-looking number next to it.

### Where this fits Mayo's own process
This is the same subgroup sensitivity/specificity/bias breakdown
Platform_Validate performs as part of Solutions Studio qualification,
and the same shape of evidence Mayo's own qualified solutions have
reported (a comparable fairness-aware clinical AI system reduced a
subgroup detection gap from 30.3 to 7.4 percentage points through this
exact kind of measure-mitigate-remeasure cycle) — reusing the identical
methodology this project already built means Notebook 3's output is
already in the shape Mayo's qualification process expects, not something
that needs reformatting later.

---

## What ties all three together

Every notebook produces output in the same shape the next one consumes:
Notebook 1's cohort is exactly what `training/omop_adapter.py` +
`LabelValidator` already know how to validate; Notebook 3's subgroup
audit is exactly what `fairness/bias_audit.py` already knows how to run.
The only genuinely new work per notebook is the Mayo-specific data
extraction (SparkSQL queries, imaging access, whatever Notebook 2's two
open questions resolve to) — everything downstream of that was built and
tested against synthetic data earlier in this project specifically so it
would not need to change when real data arrives.
