"""
CardioAI Pro — Central Orchestrator
=====================================
This is the "brain" referenced in the product brief: a single coordinator
that owns the pipeline definitions, dispatches work to agents in order,
records every hop to an in-memory event log (fanned out to WebSocket
subscribers for the live dashboard), and exposes agent health.

Design choice: pipelines are declared as ordered lists of agents rather
than letting agents call each other. That makes the whole system's control
flow readable in one place and means adding a new modality is a one-line
pipeline registration, not a change scattered across agents.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any

from orchestrator.agents import (
    AgentMessage, IngestionAgent, QualityAgent, FHIRAgent, DICOMAgent,
    InferenceAgent, DiagnosticAgent, AutomationTierAgent, ContinuumAgent,
    PatientIntakeAgent, PatientRegistryAgent, ClinicalReportAgent, ClaimsAggregatorAgent,
    LongitudinalAgent, FusionAgent, AlertAgent,
)
from ingestion.quality import DataQualityEngine
from integrations.fhir import FHIRBuilder
from integrations.dicom import DICOMService
from inference.models import ModelRegistry
from inference.automation_tiers import AutomationTierEngine
from inference.multimodal_fusion import MultiModalFusionEngine
from diagnostics.diagnostic_engine import DiagnosticEngine
from orchestrator.care_continuum import CareContinuumTracker
from orchestrator.patient_registry import PatientRegistry
from reports.clinical_report import ClinicalReportGenerator
from reports.population_report import PopulationAggregator
from longitudinal.trend_engine import LongitudinalTrendEngine


class CentralOrchestrator:
    def __init__(self, event_log_size: int = 200):
        # Shared services the agents wrap
        self.quality_engine = DataQualityEngine()
        self.fhir_builder = FHIRBuilder()
        self.dicom_service = DICOMService()
        self.model_registry = ModelRegistry()
        self.automation_engine = AutomationTierEngine()
        self.diagnostic_engine = DiagnosticEngine()
        self.continuum_tracker = CareContinuumTracker()
        self.patient_registry = PatientRegistry()
        self.report_generator = ClinicalReportGenerator()
        self.population_aggregator = PopulationAggregator()
        self.longitudinal_trend_engine = LongitudinalTrendEngine()
        self.fusion_engine = MultiModalFusionEngine()
        self.last_bias_audit: dict | None = None  # cached by /api/bias-audit/run, read by the compliance report

        # Agent registry (name -> instance), used for the /agents/status endpoint
        self.agents = {
            "ingestion": IngestionAgent(),
            "quality": QualityAgent(self.quality_engine),
            "fhir": FHIRAgent(self.fhir_builder),
            "dicom": DICOMAgent(self.dicom_service),
            "inference": InferenceAgent(self.model_registry),
            "intake": PatientIntakeAgent(self.patient_registry),
            "longitudinal": LongitudinalAgent(self.patient_registry, self.longitudinal_trend_engine),
            "fusion": FusionAgent(self.fusion_engine),
            "diagnostic": DiagnosticAgent(self.diagnostic_engine),
            "automation": AutomationTierAgent(self.automation_engine),
            "report": ClinicalReportAgent(self.report_generator),
            "alert": AlertAgent(),
            "continuum": ContinuumAgent(self.continuum_tracker),
            "registry": PatientRegistryAgent(self.patient_registry),
            "claims_aggregator": ClaimsAggregatorAgent(self.population_aggregator),
        }

        # Pipelines: modality -> ordered agent names. For ecg/wearable,
        # intake -> longitudinal -> fusion now run BEFORE diagnostic, so the
        # fused (ECG + trend) score is what the diagnosis is actually based
        # on — not just the raw single-reading ECG score. See FusionAgent
        # and DiagnosticAgent for what changed and why.
        self.pipelines = {
            "ecg": ["ingestion", "quality", "fhir", "inference", "intake", "longitudinal", "fusion", "diagnostic", "automation", "report", "alert", "continuum", "registry"],
            "wearable": ["ingestion", "quality", "fhir", "inference", "intake", "longitudinal", "fusion", "diagnostic", "automation", "report", "alert", "continuum", "registry"],
            "claims": ["ingestion", "quality", "fhir", "inference", "claims_aggregator"],
            "imaging": ["ingestion", "quality", "dicom", "fhir", "inference", "intake", "fusion", "diagnostic", "automation", "report", "alert", "continuum", "registry"],
        }

        self.event_log: deque[dict[str, Any]] = deque(maxlen=event_log_size)
        self.stats = {name: {"runs": 0, "errors": 0, "flagged": 0, "avg_ms": 0.0} for name in self.agents}
        self._subscribers: list[asyncio.Queue] = []

    # ---- pub/sub for the live dashboard -----------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _publish(self, event: dict[str, Any]) -> None:
        self.event_log.append(event)
        for q in list(self._subscribers):
            if not q.full():
                q.put_nowait(event)

    def _record_stat(self, msg: AgentMessage) -> None:
        s = self.stats[msg.agent]
        s["runs"] += 1
        if msg.status == "error":
            s["errors"] += 1
        if msg.status == "flagged":
            s["flagged"] += 1
        s["avg_ms"] = round(((s["avg_ms"] * (s["runs"] - 1)) + msg.duration_ms) / s["runs"], 2)

    # ---- main entry point ---------------------------------------------------
    async def dispatch(self, modality: str, record: dict[str, Any], source: str = "api") -> dict[str, Any]:
        """Runs `record` through the pipeline registered for `modality`."""
        if modality not in self.pipelines:
            raise ValueError(f"No pipeline registered for modality '{modality}'")

        task_id = str(uuid.uuid4())[:8]
        payload: dict[str, Any] = {"modality": modality, "record": record, "source": source}
        trace: list[dict[str, Any]] = []

        self._publish({
            "type": "task_started", "task_id": task_id, "modality": modality,
            "pipeline": self.pipelines[modality], "timestamp": time.time(),
        })

        for agent_name in self.pipelines[modality]:
            agent = self.agents[agent_name]
            msg = await agent.run(task_id, payload)
            self._record_stat(msg)
            trace.append({
                "agent": msg.agent, "status": msg.status,
                "duration_ms": msg.duration_ms, "timestamp": msg.timestamp,
            })
            self._publish({
                "type": "agent_hop", "task_id": task_id, "agent": msg.agent,
                "status": msg.status, "duration_ms": msg.duration_ms,
                "timestamp": msg.timestamp,
            })

            if msg.status == "error":
                self._publish({"type": "task_error", "task_id": task_id, "agent": msg.agent, "timestamp": time.time()})
                return {"task_id": task_id, "modality": modality, "ok": False, "failed_agent": msg.agent,
                        "error": msg.payload.get("error"), "trace": trace}

            payload = msg.payload  # each agent's output becomes the next agent's input

        self._publish({"type": "task_completed", "task_id": task_id, "modality": modality, "timestamp": time.time()})
        return {"task_id": task_id, "modality": modality, "ok": True, "result": payload, "trace": trace}

    def status(self) -> dict[str, Any]:
        return {
            "agents": [
                {"name": name, **self.stats[name]}
                for name in self.agents
            ],
            "pipelines": self.pipelines,
            "recent_events": list(self.event_log)[-25:],
        }


# Singleton used by the FastAPI app
orchestrator = CentralOrchestrator()
