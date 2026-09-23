"""
CardioAI Pro — API Routes
============================
REST endpoints for ingestion, FHIR, HL7, DICOM, agent status, plus a
WebSocket that fans out both the live simulated data streams AND the
orchestrator's agent event log, so the dashboard shows real activity.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from pydantic import BaseModel

from orchestrator.orchestrator import orchestrator
from ingestion.streaming import STREAMS
from integrations.hl7 import build_adt_a01, build_oru_r01, parse_hl7_message
from integrations.dicom import DICOMService
from fairness.bias_audit import run_audit
from reports.compliance_report import ComplianceReportGenerator
from training.label_schema import MACELabelRecord, MACEEventType, LabelValidator
from inference.automation_tiers import AutomationTier

router = APIRouter()           # mounted at /api in main.py
ws_router = APIRouter()        # mounted at root in main.py, so the socket is /ws/stream (not /api/ws/stream)
dicom_service = DICOMService()


# ---------------------------------------------------------------------------
# Ingestion — the entry point for every modality
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    modality: str
    source: str = "api"
    record: dict


@router.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        result = await orchestrator.dispatch(req.modality, req.record, req.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


# ---------------------------------------------------------------------------
# Orchestrator / agent status
# ---------------------------------------------------------------------------
@router.get("/agents/status")
async def agents_status():
    return orchestrator.status()


# ---------------------------------------------------------------------------
# FHIR R4 — inspect what the builder produces for a sample record
# ---------------------------------------------------------------------------
@router.get("/fhir/sample/{modality}")
async def fhir_sample(modality: str):
    samples = {
        "ecg": {"patient_id": "P-0001", "heart_rate_bpm": 92, "qt_interval_ms": 430, "hrv_sdnn_ms": 38},
        "wearable": {"subscriber_id": "SUB-0001", "heart_rate_bpm": 88, "spo2_pct": 96, "steps": 120},
        "claims": {"member_id": "M-10001", "member_age": 61, "risk_flags_count": 3},
    }
    if modality not in samples:
        raise HTTPException(status_code=404, detail="No sample defined for this modality")
    resources = orchestrator.fhir_builder.from_record(modality, samples[modality], "sample")
    return {"modality": modality, "input_record": samples[modality], "fhir_resources": resources}


# ---------------------------------------------------------------------------
# HL7 v2.x
# ---------------------------------------------------------------------------
@router.get("/hl7/sample/adt")
async def hl7_sample_adt():
    msg = build_adt_a01("P-0001", "DOE^JANE", "CARDIO-UNIT-4")
    return {"message": msg, "parsed": parse_hl7_message(msg)}


@router.get("/hl7/sample/oru")
async def hl7_sample_oru():
    msg = build_oru_r01("P-0001", 0.42, "moderate", 60)
    return {"message": msg, "parsed": parse_hl7_message(msg)}


# ---------------------------------------------------------------------------
# DICOM / PACS
# ---------------------------------------------------------------------------
@router.post("/dicom/metadata")
async def dicom_metadata(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        meta = dicom_service.extract_metadata_from_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse DICOM file: {exc}")
    return meta


@router.post("/dicom/ingest")
async def dicom_ingest(patient_id: str, file: UploadFile = File(...)):
    """
    The real end-to-end imaging path: extracts both metadata AND actual
    pixel-level features (integrations/dicom.py's extract_pixel_features)
    from an uploaded .dcm file, then routes it through the full orchestrator
    pipeline — including FusionAgent, which surfaces the imaging features
    alongside the fused score without folding them into it (no trained
    model exists to interpret them). /dicom/metadata above never touched
    the orchestrator at all; this is what closes that gap.
    """
    contents = await file.read()
    try:
        meta = dicom_service.extract_metadata_from_bytes(contents)
        pixel_features = dicom_service.extract_pixel_features(contents)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse DICOM file: {exc}")

    record = {
        "patient_id": patient_id,
        "modality_tag": meta.get("modality"),
        "study_instance_uid": meta.get("study_instance_uid"),
        "study_date": meta.get("study_date"),
        "pixel_features": pixel_features,
    }
    result = await orchestrator.dispatch("imaging", record, source="dicom upload — real pixel data")
    return result


# ---------------------------------------------------------------------------
# Algorithmic Bias Audit Protocol
# ---------------------------------------------------------------------------
@router.get("/bias-audit/run")
async def bias_audit_run(n_per_cell: int = 60):
    n_per_cell = max(10, min(n_per_cell, 300))  # guard against pathological requests

    def predict(features: dict):
        return orchestrator.model_registry.predict("ecg", features)

    report = run_audit(predict, n_per_cell=n_per_cell)
    orchestrator.last_bias_audit = report  # read by /api/compliance/report and /api/monitoring/status
    return report


# ---------------------------------------------------------------------------
# Automation Tier Engine
# ---------------------------------------------------------------------------
class ModalityConfidence(BaseModel):
    confidence: float


class AutomationTierRequest(BaseModel):
    diagnosis: str
    icd10_codes: list[str] = []
    ecg: ModalityConfidence
    echo: ModalityConfidence | None = None
    lab: ModalityConfidence | None = None


@router.post("/automation-tier/evaluate")
async def automation_tier_evaluate(req: AutomationTierRequest):
    decision = orchestrator.automation_engine.evaluate(
        diagnosis=req.diagnosis,
        icd10_codes=req.icd10_codes,
        ecg=req.ecg.model_dump(),
        echo=req.echo.model_dump() if req.echo else None,
        lab=req.lab.model_dump() if req.lab else None,
    )
    return {
        "tier": decision.tier.value,
        "confidence": decision.confidence,
        "icd10_codes": decision.icd10_codes,
        "diagnosis": decision.diagnosis,
        "recommendations": decision.recommendations,
        "requires_human_signoff": decision.requires_human_signoff,
        "auto_escalate": decision.auto_escalate,
        "action_label": decision.action_label,
    }


@router.get("/automation-tier/samples")
async def automation_tier_samples():
    """A few canned scenarios exercising each tier, for the dashboard's test console."""
    engine = orchestrator.automation_engine
    scenarios = [
        {"label": "STEMI, high confidence (both modalities agree)", "diagnosis": "STEMI",
         "icd10_codes": ["I21.19"], "ecg": {"confidence": 97}, "echo": {"confidence": 96}},
        {"label": "STEMI code, but low confidence — should NOT auto-escalate", "diagnosis": "STEMI",
         "icd10_codes": ["I21.19"], "ecg": {"confidence": 68}, "echo": None},
        {"label": "Atrial fibrillation, high confidence, non-urgent code", "diagnosis": "Atrial fibrillation",
         "icd10_codes": ["I48.91"], "ecg": {"confidence": 92}, "echo": None},
        {"label": "Low-confidence, unlisted diagnosis", "diagnosis": "Nonspecific ST-T changes",
         "icd10_codes": [], "ecg": {"confidence": 55}, "echo": None},
    ]
    results = []
    for s in scenarios:
        d = engine.evaluate(s["diagnosis"], s["icd10_codes"], s["ecg"], s.get("echo"))
        results.append({
            "label": s["label"], "diagnosis": d.diagnosis, "icd10_codes": d.icd10_codes,
            "tier": d.tier.value, "confidence": d.confidence, "recommendations": d.recommendations,
            "auto_escalate": d.auto_escalate, "action_label": d.action_label,
        })
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# Care Continuum Pipeline
# ---------------------------------------------------------------------------
@router.get("/care-continuum/status")
async def care_continuum_status():
    return orchestrator.continuum_tracker.status()


@router.get("/care-continuum/patient/{patient_id}")
async def care_continuum_patient(patient_id: str):
    detail = orchestrator.continuum_tracker.patient_detail(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No continuum record for patient '{patient_id}'")
    return detail


class ContinuumTransitionRequest(BaseModel):
    action: str  # "treatment" | "monitoring" | "resolved"
    note: str = ""


@router.post("/care-continuum/patient/{patient_id}/advance")
async def care_continuum_advance(patient_id: str, req: ContinuumTransitionRequest):
    tracker = orchestrator.continuum_tracker
    action_map = {
        "treatment": tracker.mark_treatment,
        "monitoring": tracker.mark_monitoring,
        "resolved": tracker.mark_resolved,
    }
    fn = action_map.get(req.action)
    if not fn:
        raise HTTPException(status_code=400, detail=f"Unknown action '{req.action}' — expected one of {list(action_map)}")
    moved = fn(patient_id, req.note) if req.note else fn(patient_id)
    return {"patient_id": patient_id, "moved": moved, **tracker.patient_detail(patient_id)}


# ---------------------------------------------------------------------------
# Patient Registry — what the clinician dashboard reads from
# ---------------------------------------------------------------------------
def _enrich_with_continuum(detail: dict) -> dict:
    continuum = orchestrator.continuum_tracker.patient_detail(detail["mrn"])
    detail["care_stage"] = continuum["stage"] if continuum else "screening"
    detail["care_history"] = continuum["history"] if continuum else []
    return detail


@router.get("/patients")
async def list_patients(status: str = "active"):
    """
    status: "active" (default — the working census), "discharged", or
    "all". Defaults to active so a discharged patient doesn't linger on
    the nursing/physician worklists by default, while staying reachable
    for chart review via status=all or status=discharged.
    """
    all_patients = orchestrator.patient_registry.list_all()
    if status != "all":
        all_patients = [p for p in all_patients if p.get("status", "active") == status]
    patients = [_enrich_with_continuum(p) for p in all_patients]
    return {"patients": patients}


@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    detail = orchestrator.patient_registry.get_detail(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No patient record for '{patient_id}'")
    return _enrich_with_continuum(detail)


class AdmitPatientRequest(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: str
    room: str
    cardiologist: str
    nurse: str
    allergies: list[str] = []
    medications: list[str] = []
    payer: str


@router.post("/patients")
async def admit_patient(req: AdmitPatientRequest):
    """
    The real admission path — a nurse or physician enters what they
    actually know about a real patient. Distinct from the implicit
    creation that happens when device data arrives for an unknown
    patient_id (which seeds random demo demographics — appropriate for a
    device stream, not for a real admission where the real details are
    known at intake time).
    """
    record, error = orchestrator.patient_registry.admit_patient(
        req.patient_id, req.name, req.age, req.sex, req.room, req.cardiologist,
        req.nurse, req.allergies, req.medications, req.payer, source="manual",
    )
    if error:
        raise HTTPException(status_code=409, detail=error)
    return _enrich_with_continuum(record.to_detail())


@router.post("/patients/{patient_id}/discharge")
async def discharge_patient(patient_id: str):
    record = orchestrator.patient_registry.discharge_patient(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No active patient '{patient_id}' to discharge.")
    return _enrich_with_continuum(record.to_detail())


@router.post("/patients/{patient_id}/readmit")
async def readmit_patient(patient_id: str):
    record = orchestrator.patient_registry.readmit_patient(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No discharged patient '{patient_id}' to readmit.")
    return _enrich_with_continuum(record.to_detail())


class ImportHL7Request(BaseModel):
    hl7_message: str
    cardiologist: str = "Unassigned"
    nurse: str = "Unassigned"
    room: str = "TBD"
    payer: str = "Unknown"


@router.post("/patients/import-hl7")
async def import_patient_hl7(req: ImportHL7Request):
    """
    Real "sync from another facility" — parses an incoming HL7 ADT
    message (what a referring facility's interface engine, e.g. Mirth or
    Rhapsody, would actually send) and admits that patient here. This is
    the RECEIVING end of interop: parsing a message handed to it, not a
    live connection reaching out to query a remote facility's system —
    an actual live HIE/interface-engine connection needs real network
    access and credentials this deployment doesn't have. Uses
    integrations/hl7.py's existing parser, extended here to pull PID
    (patient identity) and PV1 (visit/location) fields specifically.
    """
    try:
        parsed = parse_hl7_message(req.hl7_message)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse HL7 message: {exc}")

    pid_segment = next((s for s in parsed["segments"] if s["type"] == "PID"), None)
    if not pid_segment:
        raise HTTPException(status_code=422, detail="No PID segment found — can't identify the patient from this message.")

    pid_fields = pid_segment["fields"]
    patient_id = pid_fields[2] if len(pid_fields) > 2 and pid_fields[2] else None
    patient_name = pid_fields[4] if len(pid_fields) > 4 and pid_fields[4] else "Unknown (from HL7 import)"
    if not patient_id:
        raise HTTPException(status_code=422, detail="PID segment has no patient identifier (PID-3) to admit under.")

    pv1_segment = next((s for s in parsed["segments"] if s["type"] == "PV1"), None)
    admit_location = None
    if pv1_segment and len(pv1_segment["fields"]) > 2:
        admit_location = pv1_segment["fields"][2]

    record, error = orchestrator.patient_registry.admit_patient(
        patient_id, patient_name, age=0, sex="U", room=admit_location or req.room,
        cardiologist=req.cardiologist, nurse=req.nurse, allergies=[], medications=[],
        payer=req.payer, source="hl7_import",
    )
    if error:
        raise HTTPException(status_code=409, detail=error)
    return {
        "admitted": _enrich_with_continuum(record.to_detail()),
        "parsed_message_type": parsed.get("message_type"),
        "note": "Age and sex weren't in the parsed PID fields used here (a real interface engine's PID segment carries them — extend the parsing above to pull PID-7/PID-8 for a production deployment) — update the chart once confirmed.",
    }


@router.post("/patients/{patient_id}/tasks/{task_id}/toggle")
async def toggle_patient_task(patient_id: str, task_id: str):
    detail = orchestrator.patient_registry.toggle_task(patient_id, task_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No patient record for '{patient_id}'")
    return _enrich_with_continuum(detail)


@router.post("/patients/{patient_id}/claim/advance")
async def advance_patient_claim(patient_id: str):
    detail = orchestrator.patient_registry.advance_claim(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No patient record for '{patient_id}'")
    return _enrich_with_continuum(detail)


@router.post("/patients/{patient_id}/diagnostic-order/advance")
async def advance_patient_diagnostic_order(patient_id: str):
    detail = orchestrator.patient_registry.advance_diagnostic_order(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No patient record for '{patient_id}'")
    return _enrich_with_continuum(detail)


# ---------------------------------------------------------------------------
# Clinician-initiated orders — the actual "a cardiologist requested this"
# trigger, distinct from data simply arriving. Only "ECG risk re-analysis"
# can be genuinely fulfilled today (it's the only modality with a real
# pipeline); everything else is honestly recorded as pending, not faked.
# ---------------------------------------------------------------------------
ANALYZABLE_ORDER_TYPES = {"ECG risk re-analysis"}
ORDERABLE_TYPES = ANALYZABLE_ORDER_TYPES | {"Echocardiogram", "Stress test", "Holter monitor", "Coronary angiography"}


class OrderRequest(BaseModel):
    order_type: str
    ordered_by: str


@router.post("/patients/{patient_id}/orders")
async def place_patient_order(patient_id: str, req: OrderRequest):
    registry = orchestrator.patient_registry
    existing = registry.get_detail(patient_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"No patient record for '{patient_id}' — at least one reading must exist before ordering analysis.",
        )
    if req.order_type not in ORDERABLE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown order type '{req.order_type}' — expected one of {sorted(ORDERABLE_TYPES)}")

    if req.order_type not in ANALYZABLE_ORDER_TYPES:
        order = registry.add_order(
            patient_id, req.order_type, req.ordered_by, status="ordered",
            result_summary="Recorded for tracking — this modality has no real data-ingestion pathway into the engine yet.",
        )
        return {"order": order, "dispatch_result": None}

    vitals = existing["vitals"]
    if vitals.get("hr") is None:
        order = registry.add_order(
            patient_id, req.order_type, req.ordered_by, status="ordered",
            result_summary="No captured vitals yet for this patient — order recorded, awaiting data.",
        )
        return {"order": order, "dispatch_result": None}

    # This is the real round trip: a clinician's request re-runs the actual
    # 9-agent pipeline against this patient's most recently captured vitals —
    # not a device feed, not a form submission, a direct clinical request.
    record = {
        "patient_id": patient_id,
        "heart_rate_bpm": vitals["hr"],
        "qt_interval_ms": vitals["qtc"],
        "hrv_sdnn_ms": vitals.get("hrv") or 40,
    }
    dispatch_result = await orchestrator.dispatch("ecg", record, source=f"clinician order — {req.ordered_by}")

    if dispatch_result["ok"]:
        prediction = dispatch_result["result"].get("prediction") or {}
        finding = dispatch_result["result"].get("diagnostic_finding") or {}
        summary = f"MACE {prediction.get('score', 'n/a')} ({prediction.get('risk_tier', 'n/a')}) — {finding.get('diagnosis', 'no finding')}"
        status = "completed"
    else:
        summary = f"Pipeline failed at {dispatch_result.get('failed_agent')}"
        status = "failed"

    order = registry.add_order(patient_id, req.order_type, req.ordered_by, status=status, result_summary=summary)
    return {"order": order, "dispatch_result": dispatch_result}


STAFF_ROSTER = [
    {"name": "Dr. R. Alvarez", "role": "Cardiologist", "dept": "Cardiology", "on": True},
    {"name": "Dr. M. Chen", "role": "Cardiologist", "dept": "Cardiology", "on": True},
    {"name": "Dr. T. Okafor", "role": "Cardiologist", "dept": "Cardiology", "on": False},
    {"name": "Nurse J. Kim", "role": "RN", "dept": "Cardiac Unit 4B", "on": True},
    {"name": "Nurse S. Dubois", "role": "RN", "dept": "Cardiac Unit 4B", "on": True},
    {"name": "Nurse P. Larsen", "role": "RN", "dept": "Cardiac Unit 3A", "on": False},
    {"name": "A. Reyes", "role": "Medical Coder", "dept": "Billing", "on": True},
    {"name": "H. Novak", "role": "Program Admin", "dept": "IT / Administration", "on": True},
]


@router.get("/staff")
async def list_staff():
    return {"staff": STAFF_ROSTER}


# ---------------------------------------------------------------------------
# Live streaming + agent activity WebSocket
# ---------------------------------------------------------------------------
@ws_router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    event_queue = orchestrator.subscribe()

    async def pump_source(name: str, generator):
        async for record in generator():
            try:
                result = await orchestrator.dispatch(record["modality"], record["record"], record["source"])
                await ws.send_text(json.dumps({"type": "ingest_result", "stream": name, **result}))
            except Exception:
                pass  # connection likely closed; the gather() below will clean up

    async def pump_events():
        while True:
            event = await event_queue.get()
            try:
                await ws.send_text(json.dumps({"type": "orchestrator_event", "event": event}))
            except Exception:
                break

    tasks = [asyncio.create_task(pump_source(name, gen)) for name, gen in STREAMS.items()]
    tasks.append(asyncio.create_task(pump_events()))

    try:
        while True:
            await ws.receive_text()  # keep the connection alive; client doesn't need to send anything meaningful
    except WebSocketDisconnect:
        pass
    finally:
        orchestrator.unsubscribe(event_queue)
        for t in tasks:
            t.cancel()


# ---------------------------------------------------------------------------
# Payer Population Report — the "population risk report" output named in
# the product's use-case material, built from every claims record the
# engine has actually processed (see ClaimsAggregatorAgent).
# ---------------------------------------------------------------------------
@router.get("/payer/population-report")
async def payer_population_report(top_n: int = 20):
    return orchestrator.population_aggregator.report(top_n=max(1, min(top_n, 100)))


# ---------------------------------------------------------------------------
# Consumer Cardiac Score — reframes a wearable-pipeline patient's record
# (already tracked in the same PatientRegistry the clinician dashboard
# reads) in consumer-facing language, rather than clinical MRN format.
# ---------------------------------------------------------------------------
@router.get("/consumer/{subscriber_id}/score")
async def consumer_cardiac_score(subscriber_id: str):
    detail = orchestrator.patient_registry.get_detail(subscriber_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No score available yet for '{subscriber_id}'.")

    tier = detail.get("risk_tier", "low")
    score = detail.get("mace_score", 0.0)

    trend_label = {"low": "Stable", "moderate": "Worth watching", "high": "Elevated — talk to your doctor soon"}[tier]
    guidance = {
        "low": "No concerning patterns in your recent readings. Keep up your regular monitoring.",
        "moderate": "Some readings are outside your usual range. Consider mentioning this at your next check-up.",
        "high": "Recent readings show a pattern worth a cardiologist's attention — we recommend sharing this with your doctor.",
    }[tier]

    return {
        "subscriber_id": subscriber_id,
        "score": score,
        "risk_trend": trend_label,
        "personalized_guidance": guidance,
        "share_with_cardiologist_recommended": tier != "low",
        "last_updated": detail.get("last_visit"),
        "disclaimer": "Wellness information only — not a diagnosis. Scoring is a placeholder heuristic, not a validated clinical model.",
    }


# ---------------------------------------------------------------------------
# Audit Compliance Report — combines the most recent bias audit with live
# agent health into one document for internal compliance review.
# ---------------------------------------------------------------------------
@router.get("/compliance/report")
async def compliance_report():
    generator = ComplianceReportGenerator()
    return generator.build(orchestrator.last_bias_audit, orchestrator.status())


# ---------------------------------------------------------------------------
# Monitoring & Control — the cross-cutting panel named in the transformation
# diagram: model accuracy, bias gap, inference latency, EHR integration
# health, all in one place instead of scattered across separate endpoints.
# ---------------------------------------------------------------------------
@router.get("/monitoring/status")
async def monitoring_status():
    stats = orchestrator.status()
    fhir_stat = next((a for a in stats["agents"] if a["name"] == "fhir"), None)
    inference_latencies = {a["name"]: a["avg_ms"] for a in stats["agents"]}

    bias = orchestrator.last_bias_audit
    if bias:
        # "Model accuracy" here is the validation cohort's baseline accuracy
        # from the most recent bias audit — the only ground-truth-labeled
        # measurement this engine has, since live traffic carries no outcome
        # labels to score against.
        accs = [d["baseline"]["metrics"] for d in bias["dimensions"].values()]
        all_acc = [m["accuracy"] for metrics in accs for m in metrics.values() if m["n"] > 0]
        model_accuracy = round(sum(all_acc) / len(all_acc), 4) if all_acc else None
        bias_gap_pp = max(d["remediated"]["sensitivity_gap_pp"] for d in bias["dimensions"].values())
    else:
        model_accuracy = None
        bias_gap_pp = None

    return {
        "generated_at": time.time(),
        "model_accuracy": model_accuracy,
        "model_accuracy_source": "last bias-audit validation cohort (no live ground-truth labels exist)" if model_accuracy is not None else "no bias audit has run yet",
        "bias_gap_pp": bias_gap_pp,
        "inference_latency_ms": inference_latencies,
        "ehr_integration_health": {
            "fhir_agent_success_rate": round(1 - fhir_stat["errors"] / fhir_stat["runs"], 4) if fhir_stat and fhir_stat["runs"] else None,
            "note": "Proxy metric — no real EHR connection exists yet to measure actual uptime against.",
        },
    }


# ---------------------------------------------------------------------------
# MACE training label validation — lets a real data team check a candidate
# training dataset against training/label_schema.py's spec (washout,
# follow-up adequacy, feature ranges, subgroup completeness) before anyone
# trains on it. Does not train anything itself.
# ---------------------------------------------------------------------------
class LabelRecordRequest(BaseModel):
    record_id: str
    patient_id: str
    modality: str
    index_timestamp: float
    features: dict[str, float]
    event_type: str | None = None
    days_to_event: int | None = None
    follow_up_days: int
    censored: bool
    days_since_prior_mace: int | None = None
    subgroups: dict[str, str] = {}


class LabelValidationRequest(BaseModel):
    records: list[LabelRecordRequest]


@router.post("/training/validate-labels")
async def validate_training_labels(req: LabelValidationRequest):
    parsed: list[MACELabelRecord] = []
    parse_errors: list[str] = []
    for r in req.records:
        event_type = None
        if r.event_type:
            try:
                event_type = MACEEventType(r.event_type)
            except ValueError:
                parse_errors.append(f"{r.record_id}: unknown event_type '{r.event_type}' — expected one of {[e.value for e in MACEEventType]}")
                continue
        parsed.append(MACELabelRecord(
            record_id=r.record_id, patient_id=r.patient_id, modality=r.modality,
            index_timestamp=r.index_timestamp, features=r.features, event_type=event_type,
            days_to_event=r.days_to_event, follow_up_days=r.follow_up_days, censored=r.censored,
            days_since_prior_mace=r.days_since_prior_mace, subgroups=r.subgroups,
        ))

    validator = LabelValidator()
    included, report = validator.validate_dataset(parsed)
    eval_set = validator.early_detection_eval_set(included)

    return {
        "parse_errors": parse_errors,
        "report": {
            "total_records": report.total_records,
            "included_records": report.included_records,
            "excluded_records": report.excluded_records,
            "exclusion_reasons": report.exclusion_reasons,
            "window_distribution": report.window_distribution,
            "subgroup_coverage": report.subgroup_coverage,
            "feature_range_violations": report.feature_range_violations,
            "issues": report.issues,
            "passed": report.passed,
        },
        "early_detection_eval_set_size": len(eval_set),
    }
