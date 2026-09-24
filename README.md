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
  api/
    routes.py                  # REST endpoints + WebSocket
  frontend/
    index.html, app.js, style.css, assets/logo.jpg   # Main Engine console
    clinician_full_dashboard.html                     # Clinician Dashboard — Command Center, Admin, Nurses, Physician, EHR, Diagnostics, Billing; reads live from /api/patients + /api/staff, falls back to a synthetic roster if the engine isn't reachable
```
