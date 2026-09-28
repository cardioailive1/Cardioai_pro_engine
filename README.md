# CardioAI Pro — Main Engine

A working backend + frontend for the CardioScan Pro use-case diagram: hospital
EHR/ECG, payer claims, and consumer wearable data flowing through one central
orchestrator, out to FHIR R4/HL7, PACS/DICOM, and model inference points.

**Status:** this is real, running, end-to-end infrastructure — not mockups.
The one thing intentionally *not* real is the AI itself: no cardiac models
have been trained yet, so inference returns a transparent placeholder
heuristic clearly labeled as such (see "What's real vs. placeholder" below).

## Architecture

```
Browser  ─────────────►  Render Web Service (single Docker container)
                             ├─ Static frontend  (served at /)
                             ├─ REST API         (/api/*)
                             ├─ WebSocket         (/ws/stream)
                             └─ Central Orchestrator (in-process)
                                   ├─ IngestionAgent
                                   ├─ QualityAgent      → ingestion/quality.py
                                   ├─ FHIRAgent         → integrations/fhir.py
                                   ├─ DICOMAgent        → integrations/dicom.py
                                   ├─ InferenceAgent    → inference/models.py
                                   └─ AlertAgent
```

Every record — from a hospital ECG, a payer's claims batch, or a consumer's
wearable — is dispatched through the **same orchestrator** to a pipeline of
agents (`orchestrator/orchestrator.py`). The orchestrator never lets agents
call each other directly; it owns the pipeline order, logs every hop, and
fans that log out over WebSocket to the dashboard in real time.

## What's real vs. placeholder

| Piece | Status |
|---|---|
| Central orchestrator, agent pipeline, event log | **Real**, running code |
| Live data streaming (simulated ECG/wearable/claims generators) | **Real** simulator — swap the generator for a Kafka/SQS/HL7 listener to go live |
| Data quality at ingestion (schema, range, completeness, drift) | **Real**, functioning checks |
| FHIR R4 resource building (Observation, RiskAssessment, DiagnosticReport) | **Real**, spec-shaped output — not yet pushed to a live FHIR server |
| HL7 v2 ADT/ORU message building & parsing | **Real** |
| DICOM metadata extraction (pydicom) | **Real** — tested against an actual `.dcm` file |
| PACS C-FIND/C-MOVE | **Scaffolded**, not runnable without a hospital PACS endpoint to connect to |
| Model inference (MACE risk, population risk, personal score) | **Placeholder heuristic** — see `inference/models.py`. Clearly not a trained model; swap it in via `load_trained_model()` |
| Algorithmic bias audit (`fairness/bias_audit.py`) | **Real** statistical framework (subgroup confusion matrices, equal-opportunity gap, threshold-based remediation) run against the engine's actual registered model. Validation cohort is **synthetic** — see the module docstring for exactly what's real methodology vs. injected demo effect |
| Automation tier engine (`inference/automation_tiers.py`) | **Real** policy logic, wired into the live pipeline as its own agent. Requires *both* a confirmed urgent ICD-10 code and ≥95% cross-modal confidence to reach URGENT — the live ECG-only stream always passes an empty ICD-10 list, so URGENT is structurally unreachable there by design; reachable via `/api/automation-tier/evaluate` once a real diagnosis+coding exists |
| Diagnostic agent (`diagnostics/diagnostic_engine.py`) | **Real** rule-based bridge from risk score to diagnosis + ICD-10, wired into the pipeline before the automation agent. Deliberately conservative: only ever assigns a generic abnormal-finding code (R94.31), never a disease-specific one — HR/QTc/HRV alone can't honestly support that |
| Care continuum pipeline (`orchestrator/care_continuum.py`) | **Real** per-patient state machine (screening → diagnostic review → care decision → treatment → monitoring → resolved), forward-only, updated automatically by the live pipeline and manually via `/api/care-continuum/patient/{id}/advance` |
| Patient registry (`orchestrator/patient_registry.py`) | **Real** — bridges per-record pipeline hops to a per-patient chart (demographics seeded deterministically per patient_id, clinical state updated live). This is what the clinician dashboard now reads from instead of its own synthetic roster |
| Clinical report formatting (`reports/clinical_report.py`) | **Real** — the transformation step that was completely missing. Formats every pipeline run's quality/risk/diagnosis/automation output into one structured, human-readable report; wired as its own agent (`report`) on the ECG/wearable/imaging pipelines |
| Multi-modal imaging inference | **Fixed a real wiring bug** — `predict_imaging()` existed but was dead code; `InferenceAgent` always short-circuited to `None` for the imaging modality regardless. Now genuinely called; still honestly returns `"awaiting_trained_model"` since no trained imaging model exists |
| Payer population report (`reports/population_report.py`, `GET /api/payer/population-report`) | **Real** aggregation of every claims record processed. Deliberately does **not** fabricate a dollar-figure MLR impact or a specific CMS Star Rating point estimate — both need real cost/claims data this engine doesn't have |
| Consumer cardiac score (`GET /api/consumer/{id}/score`) | **Real** — reframes the same patient registry a wearable subscriber's record lives in, in consumer-facing language (trend, guidance, share-with-cardiologist flag) |
| Audit compliance report (`reports/compliance_report.py`, `GET /api/compliance/report`) | **Real** — combines the most recent bias-audit result with live agent health into one document. Explicitly not a substitute for a formal regulatory audit |
| Monitoring & Control (`GET /api/monitoring/status`) | **Real**, sourced honestly: "model accuracy" comes from the last bias-audit's validation cohort (the only ground-truth-labeled measurement that exists — live traffic carries no outcome labels), "EHR integration health" is a proxy (FHIR agent success rate), since no real EHR connection exists to measure actual uptime against |
| MACE training label specification (`training/label_schema.py`, `POST /api/training/validate-labels`) | **Real** methodology and validator — defines exactly what a labeled dataset needs (outcome definition, washout, follow-up adequacy, the specific exclusion logic that makes a "30-90 days early" claim scientifically checkable rather than assumed) and validates a candidate dataset against it. Does not itself train anything, and ships with no real patient data |

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# open http://localhost:8000
```

## Deploying to Render

1. Push this **whole repo** — `cardioai-pro/`, as-is — to GitHub. Nothing
   needs to be picked apart file by file; `render.yaml` already points at
   the right build context.
2. In Render: **New → Blueprint**, point at the repo — `render.yaml` at the
   root defines the service (Docker build from `/backend`, health check at
   `/healthz`).
3. Deploy. Frontend and backend are live together at the same URL — no
   separate frontend host or CORS config needed.

**What actually gets installed**: only `backend/requirements.txt` —
`fastapi`, `uvicorn`, `pydantic`, `pydicom`, `python-multipart`,
`websockets`, and `numpy` (a genuine live-service dependency via
`integrations/dicom.py`'s pixel feature extraction and
`longitudinal/trend_engine.py`'s slope calculations — verified with a
static import check across the whole live code path, not assumed).
`backend/requirements-training.txt` (scikit-learn, torch, torchvision —
needed only for `training/` and `imaging_models/`) is deliberately
**not** installed by the Docker build; those packages aren't imported
anywhere in the live request path, and including them would add several
minutes and multiple GB to every deploy for zero production benefit.
Install `requirements-training.txt` separately wherever you actually run
the training scripts — locally, or in a Mayo Clinic Platform Workspace
notebook.

The commented-out `worker` and `database` blocks in `render.yaml` show the
scale-out path: move ingestion off the request/response cycle into a
background worker, and move the in-memory event log / FHIR resources into
Postgres, once real hospital traffic volume needs it.

## Clinician Dashboard — now wired to the live engine

## Clinician-initiated orders — the real "user request" entry point

`POST /api/patients/{id}/orders` is the piece that closes the gap between
"data arrived" and "a clinician asked for analysis." A cardiologist or
nurse places an order (`order_type`, `ordered_by`); for `"ECG risk
re-analysis"` — the only type this engine can genuinely fulfill — the
endpoint re-runs the real 9-agent pipeline against the patient's most
recently captured vitals, with `source` set to `"clinician order — {name}"`
so the trace shows the request as the actual trigger, not a device feed.
Every other order type (Echocardiogram, Stress test, etc.) is honestly
recorded as `"ordered"` with a note that the engine has no real ingestion
pathway for that modality yet — nothing is faked. Full order history is
tracked per patient and surfaced in the clinician dashboard's patient
drawer. The dashboard's "Request Analysis" panel on the Physician page is
the UI for this.


`frontend/clinician_full_dashboard.html` (served automatically at
`/clinician_full_dashboard.html` once deployed) reads real data from
`/api/patients` and `/api/staff` — patient demographics, vitals, MACE
score, diagnostic finding, automation tier, care stage, and ICD-10/CPT
coding all come from the actual pipeline, not a separate synthetic roster.
Task toggles, diagnostic-order advances, and claim advances write back to
the engine via `POST /api/patients/{id}/...`.

One honest thing to know: the engine only creates a patient record when a
record actually flows through the `ecg` or `wearable` pipeline. The
simulated streams only run once something opens the `/ws/stream`
WebSocket (see `api/routes.py`) — so if you open the clinician dashboard
by itself with a freshly started engine and nothing has ever connected to
that socket, you'll correctly see "0 patients yet," not fake data. Open
the Main Engine console's Live Data Streams page (or POST to `/api/ingest`
directly) to generate traffic for the clinician dashboard to display. If
the engine isn't reachable at all, the dashboard falls back to its own
synthetic 12-patient roster, clearly labeled as such in the banner.

## Real admission, discharge, expanded vitals, and facility sync

Real gaps a clinician using this dashboard would hit immediately, closed:

**Admit/discharge, not just implicit patient creation.** Before this,
typing any unknown patient ID anywhere in the dashboard silently created
a patient with RANDOM demo demographics ("Demo Patient P-0099", a random
age, a random allergy list) via `PatientRegistry._seed_new()` — fine for
a device stream that will never carry a real identity, clinically wrong
for a nurse who just admitted a real person and knows their real name.
`PatientRegistry.admit_patient()` is the real path (`POST /api/patients`
— Nurses page's "Admit New Patient" form, and Physician page's "Quick
Admit" for a new consult); `_seed_new()` stays as the honest fallback for
data arriving with no admission first, now explicitly labeled
`admission_source: "auto"` to distinguish it. `discharge_patient()` /
`readmit_patient()` (`POST /api/patients/{id}/discharge` /
`/readmit`) round out the status lifecycle; `GET /api/patients` defaults
to `status=active` so a discharged patient doesn't linger on the nursing/
physician worklists, while staying reachable via `status=all`.

A real bug surfaced and fixed while building this: `to_detail()` didn't
expose the new `status` field at all, so the `status=active/discharged`
filter was silently defaulting every patient to "active" regardless of
real state — found via live testing (discharge a patient, then check
whether they still show as active), not code review.

**The "Admit New Patient" form is a real 6-step EHR intake wizard**, not
a single flat form — Personal → Demographics → Location → Insurance →
Clinical → Vitals & Submit, with a stepper indicator, per-step
validation (can't advance past Personal without a patient ID and name,
past Demographics without a valid age, etc.), a live summary on the
final step, and a combined submit that admits the patient *and* chains a
real vitals-ingest call through the pipeline if any reading was entered
on step 6 — one continuous flow instead of two disconnected actions.
Demographics' race/ethnicity field deliberately uses the exact same
category strings as `fairness/bias_audit.py`'s and
`training/label_schema.py`'s `SUBGROUP_DIMENSIONS`, so a real admission
captured here doesn't need translating to feed that machinery later.

A second real bug, caught while extending the shared admit logic for the
wizard: the sex field was hardcoded to read from `"admitSex"` regardless
of which form called it, so the Physician page's "Quick Admit" — a
separate, minimal form with no sex field of its own — would have
silently submitted whatever the Nurse wizard's sex dropdown happened to
contain. Fixed to read `${idPrefix}Sex` like every other shared field.
Verified with a full jsdom run through all 6 steps (empty-step
validation blocking advancement, each step correctly marked "done",
final combined submit producing a real MACE score) and a direct
server-side check confirming every new field — phone, race/ethnicity,
unit, insurance member ID, reason for admission — actually persisted,
not just that the UI looked right.

**A third real gap, reported directly by a user rather than found in
testing**: the step indicators in the wizard's top bar looked clickable
but did nothing — `renderAdmitWizardStep()` toggled their active/done
CSS classes, but no click handler was ever attached to them. Fixed
properly rather than just cosmetically: clicking a completed step now
jumps straight there, with a separate `admitWizardFurthestStep` tracked
alongside the current step specifically so that reviewing an earlier
step doesn't lock you back out of later steps you'd already filled in —
the first version of the fix conflated "current step" with "furthest
reached," which would have made jumping back from step 3 to review step
1 visually un-complete steps 2-3 even though their data was still there.
Verified with a dedicated test: skipping ahead before validating is
still blocked, jumping back to a done step preserves that step's data,
and jumping forward again to an already-reached step works even after
navigating back past it.

**Expanded vitals — real CVD tracking needs more than HR/QTc/HRV.** BP
(systolic/diastolic), SpO2, respiratory rate, temperature, and weight are
now charted live (`vitals`/`vitals_history`), replacing the old `bp`/
`spo2` fields that were static, seeded once at patient creation, and
never actually updated from real data. Deliberately NOT added to
`ingestion/quality.py`'s ECG schema: that schema treats every field as
*required* and penalizes absence — adding these there would have scored
down every ordinary device-stream ECG record for "missing" six fields it
was never meant to carry. Built as a separate, optional range-check
(`validate_expanded_vitals`) instead, verified to catch a bad value
(temp=999) without false-flagging valid readings. Weight trend
specifically matters for heart-failure decompensation risk — tracked now,
not yet fed into the MACE score itself; a real next integration step,
named honestly rather than implied as already done.

**Facility sync via HL7 ADT import** (`POST /api/patients/import-hl7`,
Physician page's "Import from Another Facility") — parses a real ADT^A01
message the way a referring facility's interface engine (Mirth,
Rhapsody, etc.) would send one, and admits that patient here. This is the
*receiving* end of interop — parsing a message handed to it — not a live
connection reaching into another facility's system, which needs real
network access and credentials this deployment doesn't have. Verified
with a genuine round trip: generated a test message with the project's
own `build_adt_a01()` builder, fed it through the new parser-based import
endpoint, and confirmed it correctly extracted patient ID, name, and
admit location.

A second real bug, caught by testing the actual UI rather than just the
API: `fetchWithTimeout()` already parses the response body and throws
internally on a non-ok HTTP status — the admit/discharge/import handlers
were incorrectly re-checking `.ok`/`.status` on that already-parsed body
(which don't exist there), so every one of them reported failure even on
success. Root-caused by comparing server-side state (confirmed correct)
against what the UI displayed (wrongly showed an error), not assumed from
a passing test. Fixed by matching the correct, already-used pattern
elsewhere in the same file. A related second bug, found the same way:
discharging a patient from the drawer made them vanish from the
in-memory patient list entirely (since the default `/api/patients` fetch
that refreshes the dashboard is active-only), so the drawer couldn't
re-render their new state — fixed by fetching that one patient directly
after a discharge/readmit action, without changing what the worklists
show by default.

## MACE training labels — making "30-90 days early" a checkable claim

`training/label_schema.py` defines what a real training dataset needs
before "detects MACE risk 30-90 days early" is something a model has
actually been shown to do, rather than a `"window_days": 60` field sitting
in a placeholder dictionary (see `inference/models.py`).

The methodological core: a model that's simply good at detecting current
acute abnormality will trivially look like it "predicts" imminent events,
because sick-now and event-soon are correlated for mundane reasons. That's
not early detection. The guard is exclusion — `early_detection_eval_set()`
keeps only records where the event fell in the 30-90 day window (or no
event occurred with adequate follow-up), and drops any record with an
event in the first 30 days entirely, so a model can only score well on
this specific evaluation by genuinely detecting something 30-90 days out.

`LabelValidator` also enforces: a 90-day washout before the index
observation (excludes patients mid-event), minimum follow-up before a
negative label is trusted, feature-range compliance against the same
bounds `ingestion/quality.py` enforces live, and subgroup-label
completeness so a real dataset can be run through the existing bias audit
unchanged. Test it against a candidate dataset with `POST
/api/training/validate-labels` — it validates, it doesn't train anything,
and it ships with zero real patient data.

## Training pipeline — benchmark model, candidate model, real comparison

`training/train_model.py` (run it: `python -m training.train_model`) is a
full, working training pipeline built on the label schema above:
generate/load a dataset → validate it → build the early-detection eval
set → split by *patient* (not record — the same patient must never appear
in both train and test) → fit a benchmark and a candidate model → compare
them fairly.

**The benchmark** (`training/benchmark_model.py`) is a logistic regression
over the three features this engine already ingests (HR, QTc, HRV) — the
statistical floor a candidate model has to clear. It is explicitly *not*
the real clinical risk scores (HEART, TIMI, GRACE) used in cardiology
practice, which need variables (troponin, history, ECG morphology) this
engine doesn't ingest yet; the module docstring cites commonly-reported
AUC ranges for those instruments from the literature, for calibrating
expectations, not as a target this local baseline is expected to hit.

**The candidate** is a gradient boosting model, included specifically
because the synthetic hazard function has a real nonlinear interaction (HR
× HRV) a linear model can't easily capture.

**Running it against the bundled synthetic generator** (`training/synthetic_dataset.py`
— reuses the same representation-gap mechanism as the bias audit) produced:
AUC 0.707 (benchmark) vs 0.714 (candidate) — a small raw difference. But
comparing at the same fixed threshold was actively misleading given how
differently calibrated the two models were (Brier score 0.230 vs 0.072) —
so the script also reports each model's own threshold for a matched ~80%
sensitivity operating point, the fair comparison: benchmark
specificity 0.371 / PPV 0.096 vs candidate specificity 0.445 / PPV 0.108.

This proves the *pipeline* works end-to-end — data generation, validation,
patient-level splitting, training, calibration-aware evaluation. It does
**not** produce a model for the live engine: both models were trained on
synthetic outcomes. `inference/models.py`'s `load_trained_model()` stays
returning `None` until this same pipeline runs against a real,
IRB-approved, clinically-adjudicated outcomes dataset — swapping the data
source in `train_model.py`'s `main()` is the only change that requires.

## Imaging model architecture — vision transformers for echo + CT

`imaging_models/` closes the architecture gap for the imaging modality:
`vit_backbone.py` is a real, from-scratch ViT-Base backbone (85.4M
parameters, verified via an actual forward pass — not just written and
assumed correct), which `echo_model.py`'s `EchoVideoViT` and
`ct_model.py`'s `CardiacCTViT` build on top of. These aren't arbitrary
architecture choices — they follow real, published, peer-reviewed 2025/2026
papers: **DINO-LG** and **CARD-ViT** (coronary CT calcium scoring, ViT-Base/8
self-supervised pretrained with DINO, 89%/90% sensitivity/specificity on
914 CT scans) and **Echo-Vision-FM** (Nature Communications; "Echo-VideoMAE,"
a ViT masked autoencoder over echo video, 89.12% accuracy / 0.9364 AUC for
LVEF classification). Both share the same core strategy: self-supervised
pretraining on *unlabeled* imaging data — unlabeled scans are far easier
to obtain than expert-annotated ones — then a small supervised head
fine-tuned on the actual downstream task.

Verified with real forward passes on synthetic tensors: the full
`CardiacCTViT` (ViT-Base/8, 85.7M params, matching DINO-LG exactly) runs
in 1.4s on CPU; `EchoVideoViT` correctly processes an 8-frame cine loop
through per-frame spatial encoding + temporal transformer.

**Edge-deployment alternative for echo, added after checking real
precedent**: `mobilenet_echo_model.py`'s `MobileEchoNet` wraps
torchvision's standard MobileNetV3 (not hand-rolled — MobileNetV3's
NAS-derived architecture is easy to get subtly wrong by hand, and there's
no benefit to reimplementing it when no cardiac-specific change is needed
beyond the input/output layers). This isn't a blanket "MobileNet instead
of ViT" swap — it's matched to where real published precedent actually
supports it: a comparative study of five architectures for echo ejection-
fraction analysis found MobileNet was the best choice for portable
deployment, shipping it on a Raspberry Pi at ~10x faster than manual
expert analysis, with a stated path to smartphone deployment — point-of-
care/handheld ultrasound is a real, large, growing clinical use case with
a genuine compute constraint a hospital's cloud GPU doesn't have. CT stays
on the ViT path; there's no equivalent "point-of-care CT" driver, since CT
scanners are stationary hospital equipment with server-adjacent compute
already available.

Benchmarked properly (warmed up, not a cold-start single run — an
unwarmed first test was genuinely misleading due to PyTorch's cold-start
compilation overhead, caught and redone rather than reported): on the
same 8-frame input, `MobileEchoNet`-Large (8.5M params) runs **17.8x
faster** than the full `EchoVideoViT` (94.9M params); the Small variant
(2.9M params) runs **80x faster**. `load_trained_model()` picks between
the two based on deployment context — cloud/hospital vs. edge/handheld —
without any other code needing to know which one is loaded.

**What this is and isn't**: real, tested architecture code, wired into
`inference/models.py`'s `load_trained_model()` hook exactly like the ECG
model. It is NOT a trained model — no pretrained weights exist. Real DINO/
MAE self-supervised pretraining needs the datasets the cited papers used
(a large unlabeled cardiac CT or echo-video corpus) and GPU compute this
environment doesn't have. `EchoVideoViT` also simplifies Echo-VideoMAE's
joint spatiotemporal masked pretraining into per-frame spatial encoding +
separate temporal encoding — stated in that module's docstring, not hidden.

## OMOP CDM adapter — the bridge to Mayo Clinic Platform's real data

`training/omop_adapter.py` is the one genuinely new piece of code the
Mayo Clinic Platform_Discover training path (Cohort Visualizer → SparkSQL
→ Jupyter Workspace) needs — everything downstream (`LabelValidator`,
`train_model.py`) is unchanged. Takes OMOP CDM's standard PERSON /
MEASUREMENT / CONDITION_OCCURRENCE / DEATH table rows (plain dicts — no
pandas/Spark dependency, works with either) and produces
`MACELabelRecord` objects.

**Deliberately doesn't hardcode any condition/measurement concept_id.**
OMOP concept_ids are vocabulary-version and institution-specific;
guessing one from memory risks silently mislabeling real patient data
with a subtly wrong code — worse than a crash, since nothing would look
wrong. Every concept_id is a required field on `OMOPConceptMapping`, filled
in after looking them up in Mayo's own CONCEPT table. `OMOPConceptMapping.validate()`
catches an incompletely-filled mapping before it silently produces wrong
labels, verified to fire real warnings for each gap (no MACE concept_ids
at all, missing HRV, empty race mapping).

**A real gap surfaced, not hidden**: HRV is rarely a discrete, structured
OMOP Measurement at most institutions — it needs raw ECG waveform
analysis, unlike heart rate and even QTc. The adapter handles a missing
HRV concept_id by leaving `hrv_sdnn_ms` out of the feature dict, which
`LabelValidator` already tolerates.

**Verified against the existing, unmodified downstream pipeline** — this
is the part that actually proves the bridge works, not just that the
adapter runs without crashing. A synthetic four-patient test included a
deliberate washout case: a patient with a stroke 61 days after their
index reading (which alone would be a clean early-window positive) *and*
an MI 46 days before it. The adapter correctly computed
`days_since_prior_mace=46`, and `LabelValidator` — completely unmodified
— correctly excluded that record for `prior_mace_within_washout`. The
other three records (a clean early-window MI, a censored negative, and a
cardiovascular death sourced from the Death table) all validated and
classified correctly.

### Where this gets used — the Jupyter Workspace notebook flow

`omop_adapter.py` is used inside the Jupyter notebook in Mayo Clinic
Platform's Workspace, as the bridge between querying real data and
running the already-built, already-tested training pipeline. It's cell 3
of 5:

**Cell 1 — get the project code into the Workspace**
```python
# git clone your repo, or upload cardioai-pro/backend/ as a folder, then:
import sys
sys.path.insert(0, "/path/to/cardioai-pro/backend")
```

**Cell 2 — query OMOP CDM via SparkSQL, collect into plain dicts**
```python
persons = spark.sql("""
    SELECT person_id, gender_concept_id, year_of_birth, race_concept_id, ethnicity_concept_id
    FROM person WHERE person_id IN (SELECT person_id FROM your_approved_cohort)
""").collect()
persons = [row.asDict() for row in persons]

measurements = spark.sql("""
    SELECT person_id, measurement_concept_id, measurement_date, value_as_number, visit_occurrence_id
    FROM measurement
    WHERE measurement_concept_id IN (HR_MAPPED_ID, QTC_MAPPED_ID, HRV_MAPPED_ID_IF_EXISTS)
""").collect()
measurements = [row.asDict() for row in measurements]

conditions = spark.sql("SELECT person_id, condition_concept_id, condition_start_date FROM condition_occurrence WHERE ...").collect()
conditions = [row.asDict() for row in conditions]

deaths = spark.sql("SELECT person_id, death_date, cause_concept_id FROM death WHERE ...").collect()
deaths = [row.asDict() for row in deaths]
```
The actual concept_id values for the `IN (...)` filters come from Schema
Visualizer, used separately, outside the notebook, before writing this
cell — not something this adapter looks up for you.

**Cell 3 — this is where `omop_adapter.py` is actually used**
```python
from training.omop_adapter import OMOPConceptMapping, build_mace_records

mapping = OMOPConceptMapping(
    heart_rate_concept_id=...,   # from your Schema Visualizer lookup
    qtc_concept_id=...,
    hrv_concept_id=None,         # likely None — see this module's note on why
    mi_condition_concept_ids={...},
    stroke_condition_concept_ids={...},
    cv_death_cause_concept_ids={...},
    race_concept_id_map={...},   # from PERSON table's race_concept_id values in your cohort
)
print(mapping.validate())  # check this BEFORE running on the real cohort

records = build_mace_records(persons, measurements, conditions, deaths, mapping, max_followup_days=365)
print(f"Built {len(records)} records from your real cohort")
```

**Cell 4 — validate, exactly as already tested on synthetic data**
```python
from training.label_schema import LabelValidator
validator = LabelValidator()
included, report = validator.validate_dataset(records)
print(report.exclusion_reasons, report.window_distribution, report.issues)
```

**Cell 5 — train, exactly as already tested**
```python
# Same code already in training/train_model.py's main() — patient-level
# split, benchmark + candidate training, calibration-aware evaluation —
# just fed `included` from the real cohort instead of the synthetic
# generator's output.
```

Cell 3 is the seam: everything before it is Mayo-specific (SparkSQL,
OMOP schema, concept_id lookups); everything after it is already-built
code that has no idea the data came from Mayo at all — it only ever sees
`MACELabelRecord` objects, the same shape whether they came from
`training/synthetic_dataset.py`'s generator or a real cohort. That's why
the adapter is a separate module rather than folded into the query cell:
it's the one place Mayo-specific logic lives, so nothing downstream has
to change regardless of where the data comes from.

## Training loops — real, run, and honestly debugged

`training/pretrain_dino_ct.py`, `training/pretrain_mae_echo.py`, and
`training/train_mobile_echo.py` implement the three training paths the
imaging models section above describes. All three were actually run, not
just written — including finding, diagnosing, and fixing two real bugs
along the way:

**A real architecture bug**: DINO's multi-crop augmentation uses two
resolutions (224px global crops, 96px local crops) through the *same*
network — `ViTBackbone` originally assumed one fixed size, so a 96px crop
produced a different patch count than its positional embeddings were
sized for. Fixed with the standard ViT technique (bicubic interpolation
of the positional embedding grid) rather than a workaround, verified with
a regression test on the original fixed-size case plus the new
variable-size case.

**DINO pretraining (CT)** — genuine collapse was found: the loss
converged cleanly to ln(1024), exactly the entropy of a uniform
distribution over the DINO prototypes, meaning both networks learned to
output "no information." Four controlled experiments (gradient clipping
+ lower LR, removing weight decay, cleaner synthetic signal, sharper
teacher temperature) all showed the same pattern — genuine early learning
signal, then drift to collapse — pointing to small-scale training
instability (tiny batch, ~40 total steps, no momentum warmup) rather than
a logic bug. Documented honestly in the script rather than hidden.

**MAE pretraining (echo)** — far more stable, as expected from a simple
reconstruction objective with no teacher-student dynamics: loss dropped
cleanly from 0.40 to ~0.075 with no instability.

**MobileEchoNet supervised training** — a real bug, found by a
correlation check (not just watching the loss): despite low training
loss, predictions were *negatively* correlated with true targets.
Diagnosed by directly comparing `train()` vs `eval()` mode predictions on
the same input — a real gap (0.55 vs 0.32 against a true value of 0.50),
the classic BatchNorm pitfall where `eval()` mode's running statistics
haven't converged with a small batch size and few update steps. Fixing
the batch size (4 → 16, now the default) measurably helped but didn't
fully resolve it — remaining weak correlation reflects real data-volume
limits (a multi-million-parameter CNN needs hundreds of real examples,
not four dozen synthetic ones), not a remaining code bug.

## Multi-modal fusion: ECG, longitudinal trend, and imaging

Three new pieces close the gap between "single-reading risk score" and
actually using ECG, imaging, and longitudinal data as the different
signals they are:

**`longitudinal/trend_engine.py`** — every other signal in this engine
looked at one reading in isolation, which can only ever detect current
abnormality, not a developing trend. This computes real statistics
(baseline deviation, linear slope over time, volatility) from a patient's
accumulated `vitals_history` (now actually tracked in
`patient_registry.py` — it previously only kept the latest snapshot).
Requires a minimum of 4 observations AND at least 6 hours of time spread
before claiming a trend — a real numerical-stability bug surfaced during
testing where readings seconds apart produced a slope of ~62 million
units/day; below the minimum spread, slope is now honestly `None`, not a
number that looks precise but isn't.

**`integrations/dicom.py`'s `extract_pixel_features()`** — the engine
previously only ever read DICOM header metadata, never actual pixel data.
This computes real intensity/entropy/gradient statistics from the pixel
array. It is explicitly NOT a diagnostic interpretation — nothing here
says what an elevated gradient-magnitude mean means clinically. New
endpoint `POST /api/dicom/ingest` routes a real upload through the full
orchestrator pipeline; the old `/api/dicom/metadata` never touched the
pipeline at all.

**`inference/multimodal_fusion.py`** — transparently combines the ECG
score and longitudinal trend risk with disclosed, hand-set weights
(0.75/0.25, not learned). Imaging features are surfaced alongside the
fused assessment but never folded into the score — doing so would
silently launder an uninterpreted number into looking like clinical
signal.

**Update: the fused score now genuinely drives the diagnosis.** The
pipeline was reordered — `PatientRegistry` is split into
`intake_vitals()` (early: resolves patient identity, appends to reading
history) and `finalize_clinical()` (late: saves results), so
`intake → longitudinal → fusion` now run *before* `diagnostic →
automation`, not after. `DiagnosticAgent` consumes the fused score, not
the raw ECG score alone, falling back to ECG-only on a patient's first
reading (no trend data yet exists) — verified this fallback is byte-
identical to the pre-reorder behavior. `AutomationTierAgent` needed zero
code changes; it already only reads `diagnostic_finding`, so it
transparently inherits the fusion effect.

Verified end-to-end: a patient with a *moderate* current ECG reading
(raw score 0.5) but a real 10-day worsening trend (rising HR, falling
HRV) behind it — raw ECG alone stays "moderate"; fused, it correctly
crosses into "high" tier with a different diagnosis. Automation tier
correctly stayed at "recommendation," not "urgent" — the STEMI-code
safety gate holds regardless of fusion, exactly as it should. Full
regression (all pipelines, both dashboards, 15 agents) passed with zero
errors after the reorder.

## IoMT backend bridge — connecting the live consumer-facing system

`integrations/iomt_bridge.py` (`POST /api/iomt-bridge/ingest`) connects
this engine to a separate, real, independently-deployed system — the
"IoMT CardioAI Backend" live at a production URL, with its own auth,
BLE device registry, implant registration, and vendor-gateway ingestion.
That system is the consumer/device-facing front door; this engine is the
clinical intelligence behind it — quality scoring, inference, diagnosis,
automation tier, longitudinal trend, fusion.

**What's verified**: real end-to-end tests against the live orchestrator
— a high-risk reading (HR 172, QTc 535, HRV 9) correctly ran the full
13-agent pipeline and produced a real MACE score, diagnostic finding,
and automation-tier decision, with the URGENT safety gate correctly
staying closed (generic ICD-10 code, so it landed at "recommendation,"
not auto-escalated). Unrecognized reading types surface a warning in the
response instead of silently dropping. A shared-secret `X-Bridge-Api-Key`
header (`IOMT_BRIDGE_API_KEY` env var) gates this specific endpoint —
deliberately, since this is a new connection point reachable from a live
internet-facing system, unlike the rest of this project's still-open
endpoints.

**What's assumed, not confirmed**: the inbound payload shape
(`IoMTIngestRequest`) is inferred from the live system's endpoint name
and general BLE/vendor-gateway conventions — its actual OpenAPI spec
wasn't available to design against directly. A real mismatch would fail
loudly (422 from Pydantic validation), not silently, but confirm the
real schema before connecting live traffic.

**What's an open question, not built**: the live system's documented
endpoints show only `GET /alerts` and `GET /reports` — no `POST`. There's
no documented way for this engine to push a computed risk score or
diagnosis back into that system for its consumer app to display. Either
it has an undocumented write endpoint, or it's designed to pull instead
of receiving pushes, or that direction doesn't exist yet — worth
resolving with whoever maintains that deployment rather than guessing at
an endpoint that may not exist.

**A real, honest gap this surfaced**: the bridge only carries device
readings, not patient identity — a device-only patient_id that was never
admitted through the EHR wizard first gets the same auto-seeded "Demo
Patient" fallback (`admission_source: "auto"`) as any other unknown
device stream. If the live consumer app has real patient/user identity
at signup, extending the bridge payload to carry it through to a real
`admit_patient()` call (rather than the auto-seed fallback) is the
natural next step, not something this bridge does today.

**For whoever maintains the live IoMT backend** — the other side of this
bridge, added there (not built here, since this project has no access to
that deployment):

```python
import httpx

async def forward_to_cardioai_pro(device_id, patient_id, vendor, readings):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://<cardioai-pro-deployment>/api/iomt-bridge/ingest",
            headers={"X-Bridge-Api-Key": "<the shared secret>"},
            json={"device_id": device_id, "patient_id": patient_id, "vendor": vendor, "readings": readings},
        )
        return resp.json()
```

Called from wherever `POST /vendor-gateway/ingest` currently lands, once
its actual payload shape is confirmed against `IoMTIngestRequest` above.

### The outbound direction — now built on this side, and what config the IoMT backend needs

`integrations/iomt_client.py` (`IoMTBackendClient`) is CardioAI Pro's
outbound half — it can `GET /devices`, `GET /alerts`, `GET /reports`
(pull), and `POST` a computed result back (push). `iomt_bridge.py`'s
ingest handler now calls the push automatically after every reading it
processes — best-effort: if the push fails, the response says so under
`push_back_status` without failing the request that already succeeded.

**Verified two ways, not just written and assumed**: against a local
mock standing in for the IoMT backend (all four client methods —
`get_devices`, `get_alerts`, `push_clinical_result`, and the 401 case for
a wrong key — round-tripped correctly, including a full
receive-reading → run-pipeline → push-result cycle in one request), and
separately against the **real live URL**, where the push correctly failed
gracefully rather than breaking the inbound response. That real test
surfaced something worth knowing plainly: the failure came back as
**403 Forbidden, not 404 Not Found** — meaning something in front of that
path (auth middleware, a WAF, a proxy) is rejecting the request before
it ever reaches route-matching, not simply "the route doesn't exist yet."
Worth knowing before assuming the fix is just "add the route."

**Config needed on the IoMT backend side** — none of this can be done
from here, since this project has no access to that deployment:

1. **Build the results-ingest endpoint** — `POST /clinical/cardioai-pro/results`
   (or whatever path fits that codebase's existing `/clinical/` naming
   convention better), accepting exactly the payload
   `push_clinical_result()` sends:
   ```json
   {
     "patient_id": "...", "source_system": "cardioai-pro",
     "risk_score": 0.0, "risk_tier": "low|moderate|high",
     "diagnosis": "...", "icd10_codes": ["..."],
     "automation_tier": "informative|recommendation|urgent",
     "requires_signoff": true
   }
   ```
   Wire whatever it stores into the existing `GET /alerts` / `GET /reports`
   so the consumer app and clinical dashboard on that system actually
   surface it — receiving the POST alone doesn't make it visible anywhere.
2. **Whatever is returning 403 today** needs to allow this path through —
   confirm whether that's the same auth layer protecting the admin
   endpoints, a separate WAF/proxy rule, or something else before
   assuming adding the route alone will fix it.
3. **Issue CardioAI Pro a credential** for both directions — the vendor-key
   mechanism (`POST /admin/vendor-keys`) is the most natural fit if it's
   scoped broadly enough to also cover `GET /devices`/`/alerts`/`/reports`
   and the new results endpoint, not just `/vendor-gateway/ingest`
   specifically; if it's narrowly scoped, a separate service-account
   credential is cleaner than routing a service-to-service integration
   through the email/password `/auth/login` + refresh-token cycle built
   for human users.
4. **Confirm the real GET response shapes** — `get_devices()`/`get_alerts()`/
   `get_reports()`'s field names on this side are inferred from the
   endpoint names alone, the same honest caveat as the inbound payload
   shape above.
5. **Point CardioAI Pro at the right URL and key** — set `IOMT_BACKEND_URL`
   (defaults to the real production URL already) and `IOMT_BACKEND_API_KEY`
   as real deployment environment variables once the credential from step 3
   exists.

## PMPM payer relationship — the four gaps closed

Four genuinely different pieces, all real, tested, and wired together —
closing every gap flagged when the payer use cases were first designed.

**Real X12 837 claims parsing** (`integrations/x12_837.py`) — the actual
EDI format payers/clearinghouses exchange, not the simplified
`{member_age, risk_flags_count}` shape the claims pipeline only accepted
before. Reads the ISA envelope itself to detect the real element
separator and segment terminator a file uses, rather than assuming `*`
and `~` — a parser that assumed defaults would silently misparse any
real file using different delimiters. Sequential context tracking (not
full formal HL-loop hierarchy resolution — stated honestly in the
module) correctly handles the realistic case. Verified against a
byte-precise, realistic two-claim test file: correctly extracted member
identity, DOB-derived age, real dollar charge amounts ($4,500 and $820),
and correctly classified 2 of 3 diagnosis codes as cardiovascular-relevant
(unstable angina, systolic heart failure — diabetes correctly excluded)
using real ICD-10 chapter ranges, not an arbitrary count.

**Cost/outcomes linkage** (`reports/cost_avoidance.py`) — real
methodology for turning claim cost history into an honest cost-avoidance
estimate: fits a linear trend on each member's claims before and after
they were first flagged high-risk, compares the two slopes. Explicitly
NOT a controlled estimate (no control group — stated in the module,
not glossed over) and explicitly data-gated: needs at least 3 claims
before AND after the flag date per member, and at least 10 members with
sufficient data before reporting anything population-level. A real bug
was found and fixed while testing this: the initial slope calculation
used raw Unix timestamps (seconds) as the time axis but labeled and
displayed the result as dollars-per-*day* — a genuine ~$12.50/day trend
was rounding to $0.00 at display precision. Caught by constructing a
deliberately rising-cost test case and noticing the reported slope was
implausibly zero, not by inspection.

**PMPM contract & billing administration** (`billing/contracts.py`) —
real contract records, membership reconciliation, and invoice generation
— a commercial/legal construct deliberately kept separate from the
risk-stratification population report. Reconciliation compares the
*contracted* member count (negotiated) against the *reconciled* count
(actual claims activity), flagging when they diverge by more than 10%
rather than silently accepting either number. Invoicing supports both the
PMPM industry norm (bill per contracted covered life) and a
usage-reconciled alternative, with automatic proration when a contract's
own start/end date only partially covers the billing period — verified
against hand-computed expected values (14/29 days = 0.4828 proration
factor, matching exactly).

**A payer-facing portal** (`frontend/payer_portal.html`) — a real,
separate dashboard (population risk, claims upload, cost avoidance,
contracts & billing), served automatically alongside the other two
dashboards from the same backend, no additional wiring needed. Verified
with a full jsdom run through all four pages against the live backend:
uploading the real test 837 file, watching the population report update
from it, confirming cost avoidance correctly reports insufficient data,
creating a contract, reconciling (correctly flagging a 2-vs-3,000-member
gap), generating an invoice ($15,750 = 3,000 × $5.25, computed live, not
hand-checked after the fact), and loading invoice history — zero errors.

**A real gap in that first version, reported directly and fixed**: the
Population Risk page had no interactive elements at all — no manual
refresh, and critically, the high-risk cohort table wasn't clickable,
unlike the established pattern everywhere else in this project (the
clinician dashboard's patient drawer). Fixed with a real member-detail
endpoint (`GET /api/payer/members/{member_id}`, returning full claim
history plus that member's individual cost trend — not just the
truncated slice the top_n-limited population report carries) and a
drawer UI matching the clinician dashboard's pattern. Caught a real test
artifact while verifying the fix, not a bug: an initial test run showed
"0 clickable rows found," which traced back to the test data genuinely
having no high-risk members that session (the cohort table correctly
only lists high-tier members) — not a broken click handler. Re-verified
with a genuinely high-risk member (score 0.94) submitted specifically to
test the path: row click, drawer open/close via both the X button and
backdrop click, and a 404 case for an unknown member ID — all confirmed
against the live backend, zero errors.

## Real enrollment data — closing the claims-vs-coverage gap

A real, named gap: claims (X12 837) tell you who USED services; nothing
told this system who is actually COVERED. `contracted_member_count`
reconciliation could only ever check against members with claims
activity — a usage proxy — when PMPM billing is specifically NOT
usage-based (you pay per covered life whether or not they submit a claim
that month).

**`integrations/x12_834.py`** — real X12 834 (Benefit Enrollment and
Maintenance) parsing: additions (021), terminations (024), changes
(001), reinstatements (025), with unrecognized maintenance codes
surfaced in the response rather than silently dropped. Shares its
envelope/delimiter-detection logic with the 837 parser via a new
`integrations/x12_common.py` — extracted during this build rather than
duplicated, and re-verified the 837 parser produces byte-identical
output after the refactor before building on top of it.

**`billing/enrollment.py`** — a real `EnrollmentRegistry` tracking active
vs. terminated membership from those events, with the edge cases handled
deliberately, not glossed over: a termination for a member with no prior
recorded addition still gets recorded rather than dropped (real files
can arrive mid-history); a "change" event with no existing member is
treated as an implicit addition rather than discarding real member data.

**Reconciliation now defaults to real enrollment data** when it exists,
falling back to the claims-activity proxy only when no enrollment data
has been uploaded yet — verified with a live before/after test: before
any 834 upload, `source_used: "claims_activity"`; after uploading real
enrollment events, the same reconciliation call automatically switched
to `source_used: "enrollment"` and the reconciled count changed to
reflect only truly active members, not everyone who'd ever shown claims
activity. The Payer Portal surfaces which basis was used directly in the
reconciliation result, not just the number.

## Care management — what happens after the risk score

A real, named gap: seeing a member's risk score in the Population Risk
drawer had no tracked next step — no outreach record, no enrollment, no
referral. A human had to remember to act, and the system had no way to
know whether they did.

**`care_management/tasks.py`** — real task records (outreach call,
enrollment, referral, care coordination) with a deliberately linear
status lifecycle (`pending → contacted → enrolled/declined/referred`,
with terminal statuses that can't be reopened — a renewed outreach
attempt creates a new task instead of mutating history). Verified: the
terminal-status guard correctly rejects reopening a resolved task, and
`get_intervention_anchor()` correctly returns the earliest terminal-task
timestamp for a member, or `None` when no task has resolved yet.

**A real methodological improvement to `cost_avoidance.py`, not just a UI
addition**: that module's pre/post cost-trend comparison used to anchor
on `flagged_at` — the date a risk score crossed the threshold — as a
proxy for "when intervention started." A score crossing a threshold and
a care coordinator actually enrolling a member are different events.
`compute_member_cost_trend()` now accepts a real
`intervention_started_at` and prefers it over `flagged_at` when a care
task has actually resolved. This isn't cosmetic: in a real test case,
anchoring on the score date alone left only 2 pre-period claims
(insufficient to compute anything); anchoring on the real, later
intervention date correctly captured 6 pre-claims and 4 post-claims,
producing a genuine, well-supported trend the score-date anchor couldn't
even attempt.

**A care management worklist** (Payer Portal's new "Care Management"
page) and a "Create task" form built directly into the member drawer —
so acting on a risk score and tracking that action happen in the same
place. Status-advance actions in the worklist only offer valid next
states (a `pending` task can only move to `contacted`; a terminal task
offers none), matching the backend's lifecycle exactly rather than
duplicating that logic loosely in the UI.

**A real bug found and fixed while verifying this, not a UI cosmetic
issue**: creating a task from the member drawer worked correctly, but
navigating to the Care Management page afterward showed it as empty.
Traced to a genuine, systemic gap — clicking a sidebar nav item only
ever toggled which page's CSS was visible; it never reloaded that page's
data. Any page's content was only ever as fresh as its last manual
refresh or its own polling interval, regardless of what changed
elsewhere in the meantime. Fixed by triggering each page's real loader
function on navigation to it, not just on initial script load or a
manual refresh click — re-verified with the full task lifecycle (create
from drawer → appears in worklist → advance twice through real status
transitions → correctly disappears from a "pending" filter once
resolved) end to end, zero errors.

## Investor data room — real financial model, no fabricated figures

`frontend/data_room.html` — a fourth dashboard, password-gated, built
directly from the attached three-year financial model (FY2027–FY2029),
for CardioAI Pro specifically (not any other product line).

**Every headline figure is traceable to a specific cell in the source
workbook** — read via `openpyxl` with `data_only=True` for exact values,
not the rounded preview a quick look would give. $184.1M 3-year revenue,
$63.5M 3-year EBITDA, 42.6% Year-3 EBITDA margin, $51.0M post-money
valuation — all computed directly from the "Revenue Model" sheet, which
is internally self-consistent (Gross Profit − OpEx = EBITDA, exactly,
every year).

**A real, separate reference data room existed for this project before
this one, with entirely different, unreconciled numbers** ($10.26B
5-year revenue, a $5M convertible note, $187B TAM — none of it present
in the actual financial model). Those figures are deliberately absent
here — verified directly (`!doc.body.innerHTML.includes('10.26B')` and
equivalent checks, all passing) rather than assumed absent.

**Two real data-quality issues found in the source workbook while
reading it precisely, disclosed on the Financial Model page rather than
silently resolved**: the KPI Dashboard sheet contains literal `#REF!`
formula errors in several "actual" cells (confirmed via
`openpyxl(data_only=True)`, not a parsing artifact) — the data room uses
the clean Revenue Model sheet's figures instead of those broken cells.
Separately, the OpEx Breakdown sheet's own itemized Year-1 total
($11.77M) doesn't reconcile with the $14.65M figure the P&L actually
uses for its EBITDA calculation — both numbers are shown, with the
mismatch stated plainly rather than picked silently.

**The Product & Technology section is the real differentiator this
project has that a generic pitch deck wouldn't**: it maps each of the
model's three revenue segments to the actual, tested capability behind
it — Enterprise Health Systems to the EHR admission wizard and
admit/discharge workflow, Health Insurance Payers to the real X12
837/834 EDI parsing and PMPM billing, Consumer Subscribers to the
personal cardiac score API — all real, built, and tested earlier in this
project, not aspirational.

Verified with a full jsdom run: password gate correctly blocks/unblocks,
all 6 pages navigate via both the sidebar and index document cards, the
Financial Model page's disclosures render correctly, and logout
correctly re-locks and clears the password field — zero errors.

## Document Room — real due-diligence document storage

`data_room/documents.py` — real upload/list/download/delete for
corporate, legal, HR, IP, financial, commercial, regulatory, and
insurance documents, closing a real gap: the data room previously had no
way to actually store the articles of incorporation, bylaws, or employee
contracts a real due-diligence process needs.

**Stated plainly, not buried, given the project already learned this
lesson once the hard way**: this inherits the exact same ephemeral-
storage limitation every other registry in this project has. Render's
filesystem doesn't survive a redeploy, so files uploaded here vanish the
same way the patient/contract/enrollment data did before persistence
became a known, disclosed gap. Every API response carries a
`storage_warning` field saying so, and the Document Room page surfaces
it as a banner, not a footnote — losing a signed legal document before
an investor sees it would be a real problem, not just an inconvenience.

**Two real mistakes caught and fixed while building this, not after**:
a unit mismatch where the file-size cap was set in MiB (1024²) but the
error message computed MB (10⁶), so a "25MB limit" displayed as "26MB"
— fixed to use decimal MB consistently. Separately, while wiring the new
import into `routes.py`, an edit accidentally *overwrote* the existing
X12 834 import instead of adding alongside it — caught immediately by
re-checking the import list before testing further, not discovered
later as a mysterious regression.

Verified end to end against the live backend: upload, list, and
download round-tripped byte-for-byte identical content; an invalid
category was correctly rejected; delete-then-download correctly 404s.
The UI applies the same navigation-triggered-reload pattern the Payer
Portal needed a real bug fix to arrive at — built in from the start here
instead of rediscovered.

## Business Plan — rebuilt to a standard investor due-diligence structure

The data room's Business Plan page was a thin, four-paragraph section;
it's now a full 15-section structure (Executive Summary, Company
Overview, Market Opportunity, Product & Technology, Business Model,
Go-to-Market, Competitive Positioning, Management Team, Financial Plan
Summary, Funding Request, Risk Factors, Regulatory Strategy, Social
Impact, Milestones, Appendix) — every figure and claim in it already
established elsewhere in this data room, nothing new introduced.

**What was deliberately left out, and why**: no competitor-by-competitor
matrix (no real competitive research exists to build one honestly), no
fabricated management bios beyond the one confirmed name and title, and
no total-addressable-market figure (none exists in the underlying
financial model, which is a bottom-up build from contract/member/
subscriber counts, not a top-down market-share assumption). A Risk
Factors section states the clinical-validation and regulatory-clearance
gaps directly, in the same terms used everywhere else in this data
room, rather than softening them for an investor-facing document.

## Data Room Index — a genuinely live checklist, not a static page

A new 8-category due-diligence checklist page in `data_room.html`,
structurally modeled on a reference index site but populated honestly —
"Complete" only where real content actually exists in this data room,
"Partial" where something exists but is incomplete, "Not available"
where nothing has been built (no fabricated TAM/SAM/SOM, no invented
competitor matrix, no term sheet for what is actually a priced round).

**The Legal & Corporate section is live-checked against the real
Document Room, not hardcoded** — each row queries
`GET /api/data-room/documents` and matches against actual uploads by
category (and, for items like Bylaws vs. Articles of Incorporation,
by filename/description keyword so two different document types in the
same category don't get conflated). The header stats (total items,
complete count, coverage %) are computed by counting the actual
rendered status pills on the page, not a static "45/50" figure.

**Verified this is genuinely dynamic, not just styled to look that
way**: before any upload, Articles of Incorporation correctly shows "Not
uploaded." After uploading a real document via the API and navigating
back to the page, it correctly flips to "Complete" with the real
filename shown — and, critically, Bylaws (a different document in the
same category) correctly stays "Not uploaded," proving the match logic
distinguishes between document types rather than marking an entire
category complete off one unrelated upload. Complete count and coverage
percentage both recomputed correctly after the change.

## Data room updates — round status, clinical validation, and full team bios

Several real content updates to the data room, plus a generalization of
the live document-tracking behavior:

**Page naming collision fixed.** Two pages were both labeled "Data Room
Index" (the landing/overview page and the new checklist page). Renamed
the landing page to "Overview" — it keeps its headline metrics, doc
cards, and summary table; the checklist keeps the "Data Room Index" name
that actually matches its content.

**Round status made explicit.** The $1.0M pre-seed is now shown as
"Under Due Diligence & Negotiation" on both the Overview page's header
and the Investment Structure page, with an explicit status row and
callout — not shown as if already closed.

**Clinical validation status updated, carefully.** Now "In Progress at
the Mayo Clinic Accelerate program" across all three places this claim
appears (Product & Technology, Traction & Status, Risk Factors), worded
so "in progress" can't be misread as "validated" — the underlying model
is still explicitly stated as not yet clinically validated everywhere
this appears.

**Four full executive bios added** (Sampson Kontomah, Everlyn Indirangu,
Avi Patel, Galax Wormack), replacing the earlier name-only placeholder
now that first-party bios exist to include. **One inconsistency in the
bios themselves was flagged rather than silently edited**: Everlyn's
bio references "the $2.5M seed raise" — a different, later round than
the $1.0M pre-seed this data room's financial model is built around.
Presented as given, with a callout noting it likely describes a planned
subsequent round this model doesn't yet cover, rather than quietly
rewritten to match.

**Live document-tracking generalized beyond the original Legal &
Corporate section.** The checklist's live-check logic now applies to
*any* row carrying a `data-doc-category` attribute, not just the six
rows it originally covered — Cap Table (2.3) was added as a second
live-checked item to prove the generalization actually works, not just
declared to work: uploaded a real cap table file via the API and
confirmed the row flipped from "Not uploaded" to "Complete" the same
way the original Legal & Corporate rows do.

## Financing history, SAFE terms, and traction — with two claims deliberately refused

A request to add a Mayo Clinic Platform SAFE ($800K, 20% discount, $50M
cap) and a set of traction bullets included two specific claims that
were not added, on purpose, stated directly rather than quietly
dropped: a precise "96.8% AI diagnostic accuracy" figure and an
"algorithmic bias gap < 1.8 percentage points, exceeds FDA 2024 digital
health equity guidance" claim.

**Why those two specifically, and not the rest**: everything else
requested — customer interviews, signed LOIs, a pricing survey, the SAFE
terms — is either verifiable business activity or a real financial
instrument this data room has no way to independently confirm, the same
category as the team bios added earlier. The accuracy and bias figures
are different in kind: this project has direct, first-hand knowledge
that they are false. The risk-scoring model is a tested pipeline running
on a placeholder heuristic; no accuracy validation has been performed;
the bias audit module has only ever been tested against a synthetic
cohort. Adding those two figures would have made the data room
contradict itself against content already on the Product & Technology,
Risk Factors, and Traction & Status pages — not an unverifiable claim,
a false one. Verified this exclusion actually holds: the literal strings
"96.8", "1.8 percentage points", and "exceeds FDA 2024" do not appear
anywhere in the rendered page body, not just the section that was
edited.

**One real, verified fact surfaced in the process**: Octagos Health's
$43M Series B (led by Morgan Stanley Expansion Capital, July 2024) was
independently checked via web search and confirmed accurate. The
*inference* drawn from it in the original request — that it supports an
"$80 PMPM enterprise willingness benchmark" for this company — is
presented as an argument by analogy, not as an independently validated
conclusion, since that distinction matters and the two are easy to
conflate on the page.

**The SAFE is added with an explicit, unresolved reconciliation flagged,
not silently folded into the existing round.** The $1.0M pre-seed
round's implied-dilution math (Investment Structure page) does not
currently account for this SAFE's eventual conversion, and whether total
capital raised to date is $1.8M or something else depends on a
relationship between the two instruments this data room doesn't have an
answer for yet — stated as an open question in a callout, not guessed at
silently. A new Data Room Index item (2.7) tracks this as "Needs
reconciliation" rather than marking it "Complete" prematurely.

## Market sizing and competitive analysis — real research, not fabrication

Five checklist items previously marked "Not available" were built out, and
one stayed "Not available" on purpose. The distinction that decided which
was which: an unverified-but-plausible claim (market data, competitor
funding) can be built responsibly by researching real, citable sources;
a claim about something that has never happened (a customer case study
for a company with zero customers) cannot be built at all without being
fiction.

**Market Size & Opportunity, Competitive Analysis, and Market Trends are
now real**, built from actual web research rather than declined outright
this time: 397 US health systems (American Hospital Association, 2025)
and 35.2M Medicare Advantage enrollees (KFF, February 2026) are real,
dated, sourced figures — multiplied against this company's own stated
pricing from the financial model, not a separately invented number. Four
real, named competitors (HeartFlow — $855.8M raised, IPO'd August 2025
at $1.5B; Cleerly — $385M raised, ~$729M valuation; Eko Health — $165M
raised; Octagos Health — the $43M figure already verified earlier) are
presented with their actual funding figures, each cited to a specific,
dated source, with a companion callout stating plainly that all four are
further along commercially than this company is today — not overclaiming
a competitive advantage that hasn't been earned yet.

**Consumer TAM is deliberately not reduced to a single dollar figure.**
48.6% of US adults have some form of cardiovascular disease (AHA, 2025)
— multiplying that population by a subscription price produces a number
technically defensible but practically misleading, since realistic
consumer conversion is a small fraction of anyone "at risk." The
model's own Year 3 target (33,600 subscribers) is the only consumer
figure grounded enough to stand behind, and is what's actually used.

**Customer Case Studies stays "Not available," and the reason is stated
directly on the checklist itself, not just in this README**: a case
study describes a real customer's real experience, and this company has
zero customers. There is no honest version of this item that isn't
either empty or fabricated — so it stays empty, with the reason spelled
out in place rather than silently left blank.

Marketing Materials (a one-page overview) and Partnership Strategy
(target categories, explicitly marked as not-yet-signed) were built
using only facts already established elsewhere in this data room — no
new claims introduced to fill them out.

A new "Market & Competition" page holds all of this; the Business Plan's
Market Opportunity and Competitive Positioning sections were updated to
point to it rather than continuing to state that no such research exists.

## Advisory board, FAQ, nav reorder, and deck reconciliation

**The uploaded investor deck turned out to be byte-for-byte identical**
to the Slides artifact built earlier — same 13 slides, same figures, even
the same speaker notes. Nothing needed reconciling factually. It did
surface one real, useful update though: the Data Room Index's "Company
Presentation" item had been marked "Not available" before the deck
existed — now that it's real and confirmed, that changed to "Complete."

**Advisory board bios, built from real LinkedIn/professional research,
with two treated very differently based on how confident that research
actually left me.** Dr. Tamanna Nahar and Oleg Feldgajer are both
well-corroborated across many independent, real sources — real
credentials, cited directly. Dr. Dominic Merante is a **name collision**:
multiple distinct real people share that name on LinkedIn, including an
MD (the closest match to the given title) and an unrelated logistics
company owner. Rather than pick one and present it with the same
confidence as the other two, that bio is explicitly marked "unconfirmed
— name ambiguity," with a callout stating directly that search alone
can't resolve which real person this is — attributing a stranger's real
credentials to this company's advisory board would be a genuine harm,
not just an incomplete checklist item. The Data Room Index reflects this
honestly too: Advisory Board is marked "Partial," not "Complete."

**FAQ for Investors** — five direct questions, answered using only facts
already established elsewhere in this data room (the SAFE reconciliation
gap, the LOIs' non-binding status, the aggressive Y1 sales assumption) —
no new claims manufactured to fill out a FAQ format.

**Data Room Index moved to the second nav position**, immediately after
Overview, per direct request — the checklist is now the first thing a
reviewer sees after the landing page, not buried near the end.

## Three real corrections and a full unit-economics build-out

**Dr. Merante's identity resolved, not just asserted.** With the real
LinkedIn URL confirmed directly, the Advisory Board bio dropped its
"unconfirmed — name ambiguity" flag and the Data Room Index's item 6.2
moved from "Partial" to "Complete" — the earlier caution wasn't
performative; it genuinely lifted the moment real confirmation existed,
the same way it would have stayed in place without it.

**The SAFE/pre-seed relationship — a real fact updated, not a flag
quietly removed.** The long "needs reconciliation" callout is gone, per
direct instruction — but what actually changed underneath it is real:
the Mayo Clinic Platform SAFE is signed and completed, a closed
financing event; the pre-seed remains open and under due diligence. The
two are sequential, independent instruments — a combined cap table
becomes relevant only once the pre-seed actually closes, not before.
That's a genuine resolution of the ambiguity, not the same open question
with quieter wording; the Financing History and FAQ sections, and Data
Room Index item 2.7, all reflect the same real status now.

**Full CAC/payback built for Enterprise and Payer, computed from real
model inputs — not invented multipliers.** Enterprise CAC ($213,333 per
contract, from the model's own $3.2M Y1 AE cost ÷ 15 contracts closed)
implies a 4.6-month payback and 10.5x LTV:CAC. Payer CAC ($400,000 per
contract) implies a 1.4-month payback and 34.3x LTV:CAC — a real,
computed reflection of how much revenue one large payer contract
aggregates relative to its acquisition cost, not a hand-picked number.

**Consumer CAC/LTV is honestly left uncomputed, with the specific reason
stated rather than papered over with a plausible-sounding estimate.**
The model's promotional budget is blended across segments, not split
out as consumer-specific spend, so a clean CAC numerator doesn't exist
without inventing a split the source data doesn't support — and a
multi-year consumer LTV needs a retention/churn assumption the model
never states. What's real and shown instead: consumer gross profit per
subscriber, $224.91/year, an annual figure that doesn't require an
assumption the data room doesn't have.

## Real cap table — one major discrepancy surfaced, not resolved silently

A real, uploaded cap table (CARDIO_AI_CAP_TABLE_V5A.xlsx) closed the
Cap Table checklist gap with genuine ownership data — but reading it
carefully surfaced something more consequential than the gap it closed.

**Every page in this data room stated $50.0M pre-money / $51.0M
post-money for the current raise, sourced from the financial model's
Assumptions sheet. The cap table, for the same round, shows $28,717,392
pre-money / $29,717,392 post-money — and that figure's own internal math
reconciles exactly** (pre-money + $1.0M new money = post-money, to the
dollar). $50M does appear in the cap table, but as the SAFE's own
valuation cap — a different, unrelated figure that happens to share the
number. The likely explanation: the SAFE's cap got copied into the
round's pre-money cell somewhere upstream in the financial model. That's
an inference, not a confirmation, so no headline valuation figure was
changed based on a guess — a prominent callout states the discrepancy
plainly and asks for it to be resolved directly, right on the Investment
Structure page where it's most consequential, not buried in a footnote.

**Two smaller real corrections, from what's now the more authoritative
source**: team member names — Ndirangu, not Indirangu; Womack, not
Wormack — corrected everywhere across the data room, sourced from the
formal cap table rather than the earlier bio text.

**The SAFE's derived cap price was replaced with a real number.** The
conversion price ($1.60) and conversion shares (500,000) were already
confirmed identical between the financial model and the cap table — real
corroboration. But the cap price had been an estimate (~$3.70/share,
assuming ~13.5M shares); the cap table's actual fully-diluted share
count (14,858,696) gives a real, computed $3.37/share instead.

**The Cap Table checklist item is genuinely live-checked, not hardcoded
to "Complete" because a file existed somewhere.** The real file was
uploaded through the actual Document Room API and verified to flip the
checklist row automatically — same live mechanism already proven on
other categories, now proven again on a new one with a real financial
document.

**The re-uploaded investor deck was identical to the existing one — in
fact, older than it**, still showing the SAFE as "needs reconciliation"
rather than the "signed & completed" status already applied. Nothing
new to reconcile from it this time; flagged directly rather than quietly
accepted as containing real changes it didn't have.

## Key employee update — Sampson (Chief AI Officer) and Avi (VP Web Applications)

Both title changes were grounded in real research before being written,
not just applied as given text — the same discipline as every prior
person-related update in this data room.

**Sampson Kontomah** — verified via his real LinkedIn
(linkedin.com/in/sampson-kontomah-67973988) and Crunchbase's founder
record before adding "Chief AI Officer." The new CAIO paragraph was
written fresh, not copied from Avi's bio — the original draft reused
Avi's paragraph word-for-word, which was caught and corrected on
request. His education line was corrected then re-corrected: Crunchbase
lists a B.S. from Ohio State, which was initially treated as replacing
"MBA, National University" — both are real; LinkedIn just hasn't been
updated with the MBA yet, per direct confirmation, so both degrees are
now listed together rather than one overwriting the other.

**Avi Patel** — this name is common enough that a generic search returns
dozens of unrelated people (the same problem "Dominic Merante" ran into
earlier). This one resolved cleanly: a LinkedIn profile explicitly tied
to "CardioAI Corp," describing himself as a "Full-Stack Engineer... web
development, UI/UX design" — and the company's own official team page
(cardioailive.com/team) has a real, published bio for him as VP of
Engineering. His new bio is built from that real source, not invented.

**A finding surfaced along the way, not asked for but too significant to
sit on**: the same Crunchbase record lists Tamanna Nahar as a
co-founder with the title "Executive President & Chief Medical
Director" — not merely an advisor, which is how she's currently listed
on the Advisory Board. Flagged directly in a callout on the page rather
than silently changed (outside the scope of what was asked) or silently
dropped (too consequential to leave for later discovery).

## Sampson's bio polished, and a stale callout retired

Three direct edits, no research needed this time — B Corporation
corrected to C Corporation (a factual governance-structure correction
from the person who'd know), fundraising/capital budgeting/operations
management added to his stated expertise, and the paragraph rewritten
in a more polished, professional register befitting a CEO & Chief AI
Officer bio, while keeping every real fact intact (both degrees, 10+
years, the real areas of domain expertise).

**A leftover callout from two turns back was also removed, correctly.**
It had warned that Avi Patel's bio was "word-for-word identical" to
Sampson's — true when it was written, but stale the moment Avi's bio was
rewritten with real sourced content shortly after. Leaving a warning
about a problem that no longer exists is its own kind of inaccuracy;
removed on request rather than defended.

## Business Plan cleaned of meta-commentary; CAIO scope expanded with a verified link

**Two sourcing/reconciliation callouts removed from the Business Plan
page on request** — the Everlyn $2.5M seed-raise note and the Sampson
education/Nahar co-founder note. The underlying facts those callouts
carried aren't lost: both are restated here so they're not silently
dropped just because the document no longer shows them. Everlyn's bio
still says "$2.5M seed raise," a different figure than the $1.0M
pre-seed this data room's financial model covers — unresolved. The
Crunchbase record still lists Tamanna Nahar as a co-founder with the
title "Executive President & Chief Medical Director," not merely an
advisor — also still unresolved. A prose sentence in Risk Factors
("Stated directly, not softened...") was checked and correctly left
alone — that's substantive business-plan content, not a sourcing aside.

**Sampson's Chief AI Officer scope now includes LLM development,
citing a real, checked link** — huggingface.co/nexgen2/nexgen-flash-lora,
a real LoRA fine-tune adapter on Qwen 3.5-9B. Worth knowing plainly: the
model card itself is entirely undocumented (every field reads "[More
Information Needed]," including developer/author), and nothing on the
page ties it to Sampson by name — the link is cited because it's real
and was directly confirmed as his work, not because the page itself
proves authorship.

## Premature $2.5M reference removed; Sampson's bio simplified and CAIO scope expanded

**Everlyn's bio no longer references "managing the use of funds across
the $2.5M seed raise."** That round hasn't closed — the claim was
premature, not just unreconciled with the $1.0M pre-seed elsewhere. This
also resolves what was flagged as an open discrepancy just one turn
earlier: removing the premature reference means there's no longer a
mismatch to reconcile, a cleaner fix than a caveat would have been.

**All education information removed from Sampson's bio** — both the MBA
and the Ohio State B.S. that had been added the turn before, on direct
request.

**The Chief AI Officer paragraph substantially expanded**, but only with
scope already real and documented elsewhere in this data room, not new
unverified claims: the 15-agent orchestration pipeline, the algorithmic
bias audit protocol, the imaging model architecture, the Mayo Clinic
Accelerate partnership, and the FDA 510(k) path — all things this data
room already describes as real, now explicitly placed under CAIO
oversight rather than left implicit.

## Two data-quality disclosures removed — source file corrected, per direct confirmation

The Financial Model page's callout about the KPI Dashboard's `#REF!`
formula errors and the OpEx Breakdown reconciliation gap was removed on
request — the underlying issues were corrected directly in the source
workbook. This relies on that confirmation rather than a fresh
independent re-check of the corrected file; if a verified re-check is
wanted, re-uploading the corrected workbook would let the same
`openpyxl(data_only=True)` verification used to originally find these
issues confirm the fix directly, the same rigor applied the first time.

**Three other places referenced the removed callout, and would have
been left dangling if only the callout itself were deleted**: an NRR
table row that pointed at "see data-quality note," a Financial Model
summary line on the Business Plan page that named "the two
source-workbook data-quality notes," and a Risk Factors bullet
describing "data-quality issues exist" and pointing at the same removed
section. All three were found and fixed in the same pass — removing a
disclosure without checking what else references it would have left the
document quietly broken in three other places instead of one.

A separate, different issue — the Investment Structure page's
$50.0M-vs-$28.7M pre-money valuation discrepancy — was correctly left
untouched; it's unrelated to what was reported fixed here and remains a
real, open question.

## The pre-money discrepancy resolved — $28.7M confirmed, $50M was always the SAFE cap

Per direct confirmation: $50.0M was never the round's pre-money — it's
the Mayo Clinic Platform SAFE's own valuation cap, a real, separate
figure this data room had incorrectly been repeating as the round's
pre-money everywhere. $28.7M ($28,717,392, matching the cap table
exactly) is confirmed correct.

**Every page updated in one pass, not patched one at a time**: the
Overview hero metrics, the highlight strip, the Executive Summary, the
Business Plan's Company Overview and Funding Request sections, the
Financial Model's Key Assumptions table, the Investment Structure page's
terms table (including recomputing implied dilution from 1.96% to the
correct 3.37%), the FAQ, and the Cap Table section's own resolution
callout — eight separate locations, found via a systematic sweep for
every "$50.0M"/"$51.0M"/"1.96%" occurrence, not by memory of where the
figure had been mentioned.

**What was deliberately left untouched**: every place $50,000,000
correctly refers to the SAFE's own valuation cap (the SAFE terms table,
the cap price calculation, the cap table ownership row) — those were
never wrong, and a blind find-and-replace would have broken them right
alongside fixing the real error. Verified directly, not assumed: the
SAFE's $50M cap and $3.37 cap-price calculation still read correctly
after every other instance was corrected.

## Cloudflare R2 persistence — the ephemeral-storage gap actually closed

`data_room/documents.py` now has two backends, chosen automatically:
local disk (the original, ephemeral behavior, unchanged as the default)
and Cloudflare R2 (S3-compatible), used the moment
`R2_ACCOUNT_ID`/`R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`/`R2_BUCKET_NAME`
are all set — no code change needed to switch over, and nothing breaks
for anyone who hasn't configured it yet.

**Why metadata lives in R2 too, not just the file bytes**: a naive swap
would store file content in R2 but keep the document list (filename,
category, description) in the same in-memory dict as before — fixing
half the problem and silently losing the other half on the next
redeploy. Metadata is stored as real S3 object metadata on every upload;
`list_all()` queries the bucket directly rather than a cache, so a fresh
process reflects exactly what's actually there.

**Tested without real Cloudflare credentials, and the gap in that
testing is stated directly rather than glossed over**: `moto`'s S3 mock
doesn't cleanly support arbitrary custom endpoint URLs (confirmed by
testing it directly — real 403s from a config that should work,
isolated down to the custom-endpoint-plus-`region_name="auto"`
combination specifically). So the actual logic this module implements
— metadata round-trip, content round-trip, and critically, a **fresh
client instance seeing the exact same data a prior instance wrote**
(the specific property that makes this genuinely persistent, not just
"working locally") — was verified against moto's natively-supported S3
setup instead. The `endpoint_url` and `region_name="auto"` values
themselves are copied directly from Cloudflare's own official R2
documentation, already verified via search, not something this test
needed to re-prove. A live round-trip against a real R2 bucket is the
one thing this hasn't been checked against yet.

**`render.yaml` updated** with the four R2 environment variables as
`sync: false` entries, so Render prompts for them in the dashboard
rather than expecting them committed to the repo. Re-verified in a
fresh, isolated Python environment (same rigor as the original Render
deployment check) that `boto3` — now a real, live dependency — is
genuinely captured in `requirements.txt`, not just present because it
happened to already be installed.

## Connecting real hospital systems

- **FHIR R4**: `integrations/fhir.py` builds correct resources today. Wire
  `FHIRBuilder.push()` to a hospital's Epic/Cerner FHIR endpoint with
  SMART-on-FHIR OAuth2 credentials to actually write back.
- **HL7 v2**: `integrations/hl7.py` builds/parses ADT^A01 and ORU^R01. A
  production deployment listens on a persistent MLLP socket (via an
  interface engine like Mirth or Rhapsody) rather than HTTP.
- **PACS/DICOM**: `integrations/dicom.py` extracts metadata from uploaded
  studies now. The `pacs_find()` stub shows exactly where `pynetdicom`
  C-FIND/C-MOVE calls go once there's a real PACS AE title, host, and
  network path to connect to.
- **Models**: `inference/models.py`'s `load_trained_model()` is the single
  seam where trained weights replace the placeholder heuristic, for each
  modality independently.

## Repo layout

```
render.yaml
backend/
  main.py                 # FastAPI app: mounts API + WebSocket + static frontend
  Dockerfile
  requirements.txt         # Live web service deps only (includes numpy — a real dependency, not training-only)
  requirements-training.txt # scikit-learn, torch, torchvision — for training/ and imaging_models/, not installed by the Render build
  orchestrator/
    agents.py              # IngestionAgent, QualityAgent, FHIRAgent, DICOMAgent, InferenceAgent, PatientIntakeAgent, LongitudinalAgent, FusionAgent, DiagnosticAgent, AutomationTierAgent, ClinicalReportAgent, AlertAgent, ContinuumAgent, PatientRegistryAgent, ClaimsAggregatorAgent
    orchestrator.py        # CentralOrchestrator — pipeline definitions, event log, pub/sub
    care_continuum.py      # Care Continuum Pipeline — per-patient longitudinal stage tracker
    patient_registry.py    # Patient Registry — per-record hops -> per-patient chart, what the clinician dashboard reads from
  ingestion/
    quality.py              # Data quality engine (schema/range/completeness/drift)
    streaming.py             # Simulated live ECG/wearable/claims generators
  integrations/
    fhir.py                  # FHIR R4 resource builder
    hl7.py                    # HL7 v2 ADT/ORU builder + parser
    dicom.py                  # DICOM metadata extraction + PACS scaffold
    iomt_bridge.py             # Bridge to the live IoMT CardioAI Backend — receives readings, runs the real pipeline, best-effort pushes the result back
    iomt_client.py             # Outbound client (GET devices/alerts/reports, POST computed results) — POST endpoint proposed, not yet built on their side
    x12_837.py                 # Real X12 837 EDI claims parser — reads ISA to detect real delimiters, extracts member/diagnosis/charge data
    x12_834.py                 # Real X12 834 enrollment parser — additions/terminations/changes/reinstatements, real coverage not a usage proxy
    x12_common.py              # Shared envelope/delimiter-detection logic between the 837 and 834 parsers
  inference/
    models.py                 # Model registry + inference interface + placeholder heuristics
    automation_tiers.py      # Automation Tier Engine — URGENT / RECOMMENDATION / INFORMATIVE escalation policy
    multimodal_fusion.py     # Transparent ECG + longitudinal-trend fusion — imaging surfaced, never folded into the score
  diagnostics/
    diagnostic_engine.py      # Diagnostic Agent's engine — risk score -> diagnosis + ICD-10, deliberately conservative
  longitudinal/
    trend_engine.py            # Real baseline/deviation/slope/volatility statistics from a patient's reading history
  reports/
    clinical_report.py        # Clinical report formatting — the transformation step that was missing
    population_report.py      # Payer population report — claims aggregation, no fabricated cost/Star-Rating figures
    cost_avoidance.py          # Real pre/post cost-trend methodology, data-gated — no control group, stated honestly
    compliance_report.py      # Audit compliance report — bias audit + agent health combined
  imaging_models/
    vit_backbone.py            # Real, tested ViT-Base backbone (85.4M params) — shared by both modality-specific models below
    echo_model.py              # EchoVideoViT — per-frame ViT + temporal transformer, follows Echo-Vision-FM (cloud/hospital path)
    mobilenet_echo_model.py    # MobileEchoNet — MobileNetV3 + lightweight temporal transformer (edge/point-of-care path, 17.8-80x faster)
    ct_model.py                # CardiacCTViT — ViT-Base/8 + linear head, follows DINO-LG / CARD-ViT
  training/
    omop_adapter.py           # OMOP CDM (PERSON/MEASUREMENT/CONDITION_OCCURRENCE/DEATH) -> MACELabelRecord — the Mayo Clinic Platform bridge, concept_ids supplied by you, not hardcoded
    label_schema.py           # MACE training label spec — outcome definition, washout, the 30-90-day early-detection eval-set guard
    synthetic_dataset.py      # Synthetic cohort generator for testing the training pipeline — NOT real patient data
    benchmark_model.py        # Logistic regression baseline — the statistical floor a candidate model must clear
    train_model.py            # Full training pipeline: generate -> validate -> patient-level split -> train -> calibration-aware evaluation
    pretrain_dino_ct.py       # Real DINO self-supervised pretraining loop for CardiacCTViT — documents a genuine collapse finding, not hidden
    pretrain_mae_echo.py      # Real MAE self-supervised pretraining loop for EchoVideoViT's spatial backbone — clean, stable loss decrease
    train_mobile_echo.py      # Real supervised training loop for MobileEchoNet — documents a genuine BatchNorm train/eval bug, found and partially fixed
  fairness/
    bias_audit.py            # Algorithmic Bias Audit Protocol — subgroup validation cohort, disparity metrics, threshold-based remediation
  billing/
    contracts.py               # PMPM contract records, membership reconciliation, invoice generation with automatic proration
    enrollment.py               # Real EnrollmentRegistry — active/terminated membership from X12 834 events, feeds reconciliation a real basis
  care_management/
    tasks.py                    # Care task lifecycle (outreach/enrollment/referral) — also feeds cost_avoidance.py a real intervention anchor
  data_room/
    documents.py                 # Real due-diligence document upload/list/download/delete — ephemeral storage, disclosed in every response
  api/
    routes.py                  # REST endpoints + WebSocket
  frontend/
    index.html, app.js, style.css, assets/logo.jpg   # Main Engine console
    clinician_full_dashboard.html                     # Clinician Dashboard — Command Center, Admin, Nurses, Physician, EHR, Diagnostics, Billing; reads live from /api/patients + /api/staff, falls back to a synthetic roster if the engine isn't reachable
    payer_portal.html                                 # Payer Portal — Population Risk, Claims Upload (X12 837), Cost Avoidance, Contracts & Billing
    data_room.html                                     # Investor data room — real financial model figures, password-gated
```
