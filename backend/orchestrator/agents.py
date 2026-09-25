"""
CardioAI Pro — Agent Layer
===========================
Each agent is a narrow, single-responsibility worker. The CentralOrchestrator
(see orchestrator.py) is the only thing that talks to all of them — agents
never call each other directly. This keeps the system inspectable: every
hop is a message the orchestrator can log, retry, or halt.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from ingestion.quality import validate_expanded_vitals


@dataclass
class AgentMessage:
    """The unit of work passed between the orchestrator and an agent."""
    task_id: str
    agent: str
    status: str  # "ok" | "error" | "flagged"
    payload: dict[str, Any]
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


class BaseAgent:
    """Common scaffolding: timing, error capture, uniform message shape."""

    name: str = "base"

    async def run(self, task_id: str, payload: dict[str, Any]) -> AgentMessage:
        start = time.perf_counter()
        try:
            result = await self.handle(payload)
            status = result.pop("_status", "ok")
            duration = (time.perf_counter() - start) * 1000
            return AgentMessage(task_id, self.name, status, result, round(duration, 2))
        except Exception as exc:  # noqa: BLE001 — agents must never crash the orchestrator
            duration = (time.perf_counter() - start) * 1000
            return AgentMessage(
                task_id, self.name, "error",
                {"error": str(exc), "error_type": type(exc).__name__},
                round(duration, 2),
            )

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class IngestionAgent(BaseAgent):
    """
    Entry point for every data modality (ECG waveform, wearable stream,
    claims batch, DICOM study). Normalizes shape and stamps provenance
    before anything downstream sees the record.
    """
    name = "ingestion"

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        modality = payload.get("modality", "unknown")
        record = payload.get("record", {})
        normalized = {
            "modality": modality,
            "source": payload.get("source", "unspecified"),
            "record": record,
            "ingested_at": time.time(),
            "record_id": str(uuid.uuid4())[:8],
        }
        return {"normalized": normalized}


class QualityAgent(BaseAgent):
    """Wraps the DataQualityEngine (ingestion/quality.py) as an orchestrator hop."""
    name = "quality"

    def __init__(self, engine):
        self.engine = engine

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        report = self.engine.evaluate(normalized["modality"], normalized["record"])

        # Expanded bedside vitals (BP, SpO2, resp rate, temp, weight) — a
        # separate, optional range-check (see ingestion/quality.py's
        # module comment for why these aren't in SCHEMAS itself). Adds a
        # warning without affecting the modality quality score.
        expanded_warnings = validate_expanded_vitals(normalized["record"])
        if expanded_warnings:
            report["issues"] = [*report["issues"], *expanded_warnings]

        out = {"normalized": normalized, "quality": report}
        if not report["passed"]:
            out["_status"] = "flagged"
        return out


class FHIRAgent(BaseAgent):
    """Maps a normalized, quality-passed record onto FHIR R4 resources."""
    name = "fhir"

    def __init__(self, builder):
        self.builder = builder

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        resources = self.builder.from_record(normalized["modality"], normalized["record"], normalized["record_id"])
        return {**payload, "fhir_resources": resources}


class DICOMAgent(BaseAgent):
    """
    Handles DICOM/PACS-modality records; no-ops for non-imaging records.
    If the record carries `pixel_features` (attached by the API layer when
    a real .dcm file was uploaded — see api/routes.py's /dicom/ingest —
    promotes them to the top-level payload so FusionAgent can surface them
    alongside the fused score without folding them into it.
    """
    name = "dicom"

    def __init__(self, dicom_service):
        self.dicom_service = dicom_service

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        if normalized["modality"] != "imaging":
            return {**payload, "dicom": None}
        meta = self.dicom_service.summarize(normalized["record"])
        out = {**payload, "dicom": meta}
        pixel_features = normalized["record"].get("pixel_features")
        if pixel_features:
            out["imaging_features"] = pixel_features
        return out


class InferenceAgent(BaseAgent):
    """
    Routes a record to the correct model in the ModelRegistry. Every model
    call is a stand-in inference *interface* — see inference/models.py for
    the load_trained_model() hook where real weights get dropped in.
    """
    name = "inference"

    def __init__(self, registry):
        self.registry = registry

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        if normalized["modality"] == "imaging":
            # Was silently dead before: ModelRegistry.predict() short-circuited
            # to None for imaging and predict_imaging() was never called from
            # here, so the imaging pipeline produced no prediction at all.
            prediction = self.registry.predict_imaging(payload.get("dicom") or {})
        else:
            prediction = self.registry.predict(normalized["modality"], normalized["record"])
        out = {**payload, "prediction": prediction}
        if prediction and prediction.get("risk_tier") == "high":
            out["_status"] = "flagged"
        return out


class DiagnosticAgent(BaseAgent):
    """
    Turns a risk score into an actual diagnosis + ICD-10 coding candidate —
    what AutomationTierAgent and the care continuum tracker act on
    downstream. Before this agent existed, AutomationTierAgent was
    evaluating a hardcoded placeholder string instead of anything real.
    See diagnostics/diagnostic_engine.py's module docstring for why this
    stays deliberately generic (no disease-specific codes) given the
    engine's current feature set.

    USES THE FUSED SCORE WHEN AVAILABLE, NOT THE RAW ECG SCORE ALONE: if
    FusionAgent (which now runs before this agent) produced a fused
    assessment, its fused_score/fused_tier are what this diagnosis is
    based on — not the raw single-reading ECG score. On a patient's very
    first reading, longitudinal trend data doesn't exist yet, so fusion
    falls back to ECG-only with full weight, meaning fused_score equals
    the raw ECG score exactly — behavior is identical to before until a
    patient has enough history for a real trend to exist, at which point
    it can genuinely change the outcome. See inference/multimodal_fusion.py.
    """
    name = "diagnostic"

    def __init__(self, engine):
        self.engine = engine

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        quality = payload.get("quality")
        prediction = payload.get("prediction")
        quality_score = quality["score"] if quality else 70.0

        fusion = payload.get("fusion")
        if fusion and fusion.get("fused_score") is not None:
            # Same shape as a normal prediction dict, but score/risk_tier
            # reflect the fused (ECG + longitudinal trend) assessment.
            effective_prediction = {**(prediction or {}), "score": fusion["fused_score"], "risk_tier": fusion["fused_tier"]}
        else:
            effective_prediction = prediction

        finding = self.engine.evaluate(normalized["record"], quality_score, effective_prediction)
        out = {**payload, "diagnostic_finding": {
            "diagnosis": finding.diagnosis,
            "icd10_codes": finding.icd10_codes,
            "confidence": finding.confidence,
            "specialist_review_required": finding.specialist_review_required,
            "basis": finding.basis,
        }}
        if finding.specialist_review_required:
            out["_status"] = "flagged"
        return out


class ClinicalReportAgent(BaseAgent):
    """
    Formats everything upstream (quality, inference, diagnosis, automation
    tier) into a structured, human-readable clinical report. This is the
    step that was completely missing: earlier hops produce data a machine
    can act on, but nothing turned it into something a clinician reads.
    """
    name = "report"

    def __init__(self, generator):
        self.generator = generator

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        record = normalized["record"]
        patient_id = record.get("patient_id") or record.get("subscriber_id") or normalized["record_id"]

        vitals = {
            "hr": record.get("heart_rate_bpm"), "qtc": record.get("qt_interval_ms"),
            "hrv": record.get("hrv_sdnn_ms"), "spo2": record.get("spo2_pct"),
        }
        report = self.generator.generate(
            patient_id=patient_id, modality=normalized["modality"], vitals=vitals,
            quality=payload.get("quality"), prediction=payload.get("prediction"),
            diagnostic_finding=payload.get("diagnostic_finding"), automation=payload.get("automation"),
        )
        return {**payload, "clinical_report": report.to_dict()}


class AutomationTierAgent(BaseAgent):
    """
    Decides how much autonomy the engine exercises for this finding: auto-
    escalate, queue as a clinician recommendation, or log only. Wraps
    inference.automation_tiers.AutomationTierEngine.

    Consumes DiagnosticAgent's actual finding (diagnosis + ICD-10 codes +
    confidence) rather than a hardcoded placeholder. Since DiagnosticAgent
    only ever assigns a generic abnormal-finding code (R94.31), not a
    disease-specific one like STEMI's I21.19/I21.09, URGENT stays
    structurally unreachable from this live pipeline by design — exactly
    what prevents raw vitals alone from ever auto-paging a cath lab.
    Reaching URGENT is only ever meant to happen through the
    /automation-tier/evaluate endpoint, once a real confirmed diagnosis +
    coding (from a clinician, coder, or a real trained diagnostic model)
    is in hand.
    """
    name = "automation"

    def __init__(self, engine):
        self.engine = engine

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        finding = payload.get("diagnostic_finding")
        # No automation decision to make when there's nothing to act on — a
        # high data-quality score on a "No significant finding" record is
        # confidence about nothing, not confidence in a finding, and must
        # not be read as grounds for a "recommendation" tier.
        if not finding or not finding.get("specialist_review_required"):
            return {**payload, "automation": None}

        decision = self.engine.evaluate(
            diagnosis=finding["diagnosis"],
            icd10_codes=finding["icd10_codes"],
            ecg={"confidence": finding["confidence"]},
        )
        out = {**payload, "automation": {
            "tier": decision.tier.value,
            "confidence": decision.confidence,
            "recommendations": decision.recommendations,
            "requires_human_signoff": decision.requires_human_signoff,
            "auto_escalate": decision.auto_escalate,
            "action_label": decision.action_label,
        }}
        return out


class ContinuumAgent(BaseAgent):
    """
    Final hop for the modalities that carry a patient identifier: stamps
    this record's outcome onto that patient's longitudinal care-continuum
    stage (see orchestrator/care_continuum.py). This is what turns isolated
    per-record agent hops into "where does this patient actually stand
    right now" — the view the command center and diagnostics/treatment
    screens need.
    """
    name = "continuum"

    def __init__(self, tracker):
        self.tracker = tracker

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        record = normalized["record"]
        patient_id = record.get("patient_id") or record.get("subscriber_id") or normalized["record_id"]

        self.tracker.record_screening(patient_id)
        finding = payload.get("diagnostic_finding")
        if finding:
            self.tracker.record_diagnostic_finding(patient_id, finding["diagnosis"])
        automation = payload.get("automation")
        if automation:
            self.tracker.record_care_decision(patient_id, automation["tier"])

        return {**payload, "continuum_patient_id": patient_id}


class PatientIntakeAgent(BaseAgent):
    """
    Early hop, runs right after Inference: resolves patient identity and
    appends this record's vitals to history via PatientRegistry.intake_vitals().
    Splitting registry into this early "intake" step and the late "registry"
    finalize step (below) is what makes it possible for LongitudinalAgent
    and FusionAgent to run — and their fused score to actually reach
    DiagnosticAgent — BEFORE the diagnostic decision is made, instead of
    only after. Before this split, the fused score was computed too late
    to influence anything; see FusionAgent for how that's used now.
    """
    name = "intake"

    def __init__(self, registry):
        self.registry = registry

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = payload["normalized"]
        record = normalized["record"]
        patient_id = record.get("patient_id") or record.get("subscriber_id") or normalized["record_id"]
        self.registry.intake_vitals(patient_id, record)
        return {**payload, "patient_id": patient_id}


class PatientRegistryAgent(BaseAgent):
    """
    Final hop: saves everything this pipeline run computed — quality,
    prediction, diagnostic finding, automation tier, and the longitudinal/
    fusion assessment (computed earlier by PatientIntakeAgent /
    LongitudinalAgent / FusionAgent, now genuinely upstream of the
    diagnostic decision) — onto the patient record the clinician dashboard
    reads from, via PatientRegistry.finalize_clinical(). Does not touch
    vitals_history again; that already happened in PatientIntakeAgent.
    """
    name = "registry"

    def __init__(self, registry):
        self.registry = registry

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient_id = payload.get("patient_id")
        if not patient_id:
            # Pipelines that never ran PatientIntakeAgent (e.g. claims)
            # shouldn't reach this agent at all, but fail safely if one does.
            normalized = payload["normalized"]
            record = normalized["record"]
            patient_id = record.get("patient_id") or record.get("subscriber_id") or normalized["record_id"]

        quality = payload.get("quality")
        self.registry.finalize_clinical(
            patient_id, quality["score"] if quality else 70.0,
            payload.get("prediction"), payload.get("diagnostic_finding"), payload.get("automation"),
            payload.get("longitudinal"), payload.get("fusion"),
        )
        return {**payload, "registry_patient_id": patient_id}


class LongitudinalAgent(BaseAgent):
    """
    Computes real trend statistics from this patient's accumulated reading
    history — baseline deviation, slope over time, volatility — via
    longitudinal.trend_engine.LongitudinalTrendEngine. Runs after
    PatientIntakeAgent, which already appended the current reading to
    history, so the trend computed here includes it. This is the mechanism
    a genuine "detects risk early" claim needs: every other signal in this
    pipeline looks at one reading in isolation, which can only ever detect
    CURRENT abnormality, not a developing trend.
    """
    name = "longitudinal"

    def __init__(self, registry, trend_engine):
        self.registry = registry
        self.trend_engine = trend_engine

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient_id = payload.get("patient_id")
        if not patient_id:
            return {**payload, "longitudinal": None}

        history = self.registry.get_vitals_history(patient_id) or []
        assessment = self.trend_engine.assess(patient_id, history)
        out = {**payload, "longitudinal": assessment.to_dict()}
        if assessment.trend_risk_contribution > 0.5:
            out["_status"] = "flagged"
        return out


class FusionAgent(BaseAgent):
    """
    Combines the ECG risk score and the longitudinal trend into one
    transparent fused assessment via inference/multimodal_fusion.py. Runs
    right after LongitudinalAgent and BEFORE DiagnosticAgent — this used
    to run at the very end of the pipeline, which meant the fused score
    was computed too late to ever influence the diagnosis. Now
    DiagnosticAgent consumes this agent's output directly (see that
    class), and PatientRegistryAgent persists it at the end alongside
    everything else — this agent doesn't write to the registry itself.
    """
    name = "fusion"

    def __init__(self, fusion_engine):
        self.fusion_engine = fusion_engine

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        prediction = payload.get("prediction")
        longitudinal = payload.get("longitudinal")
        imaging_features = payload.get("imaging_features")  # only ever present on the imaging pipeline

        assessment = self.fusion_engine.fuse(prediction, longitudinal, imaging_features)
        fused_dict = assessment.to_dict()

        return {**payload, "fusion": fused_dict}


class ClaimsAggregatorAgent(BaseAgent):
    """
    Rolls each claims-pipeline record into the payer population report
    (reports/population_report.py). Distinct from PatientRegistryAgent:
    this is population-level (a payer's covered members), not per-patient
    clinical state, and claims records were never wired to the registry
    since a payer's member isn't the same concept as a hospital patient.
    """
    name = "claims_aggregator"

    def __init__(self, aggregator):
        self.aggregator = aggregator

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = payload["normalized"]["record"]
        prediction = payload.get("prediction") or {}
        member_id = record.get("member_id", payload["normalized"]["record_id"])
        self.aggregator.record(
            member_id=member_id, age=record.get("member_age", 0),
            risk_flags_count=record.get("risk_flags_count", 0),
            score=prediction.get("score", 0.0), risk_tier=prediction.get("risk_tier", "low"),
            claim_id=record.get("claim_id"), total_charge_amount=record.get("total_charge_amount"),
            diagnosis_codes=record.get("diagnosis_codes"),
        )
        return {**payload, "population_member_id": member_id}


class AlertAgent(BaseAgent):
    """Final hop: decides whether a prediction should raise a clinical alert."""
    name = "alert"

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        prediction = payload.get("prediction") or {}
        alert = None
        if prediction.get("risk_tier") == "high":
            alert = {
                "level": "clinical_alert",
                "message": f"MACE risk flagged ({prediction.get('score', 0):.2f}) — route to cardiologist worklist",
                "window_days": prediction.get("window_days", 60),
            }
        return {**payload, "alert": alert}
