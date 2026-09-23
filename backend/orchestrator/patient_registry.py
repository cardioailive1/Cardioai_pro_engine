"""
CardioAI Pro — Patient Registry
====================================
Bridges the engine's per-record pipeline to a per-PATIENT view: the thing
the clinician dashboard (Command Center, Physician worklist, EHR chart,
Diagnostics queue, Billing) actually needs, instead of its own separate
synthetic roster.

Administrative/demographic fields (room, assigned staff, allergies,
medications, payer, baseline BP/SpO2) are seeded once, deterministically,
the first time a patient_id is seen — exactly the way a real system would
pull them from EHR registration rather than from the clinical data stream
itself. Clinical fields (vitals, diagnostic finding, automation tier) are
updated live by PatientRegistryAgent as records flow through the pipeline.

In-memory, same caveat as the rest of the engine: swap for real
persistence before this needs to survive a restart or run across multiple
instances.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from diagnostics.diagnostic_engine import GENERIC_ABNORMAL_DESC

CARDIOLOGISTS = ["Dr. Alvarez", "Dr. Chen", "Dr. Okafor"]
NURSES = ["Nurse Kim", "Nurse Dubois", "Nurse Larsen"]
PAYERS = ["Aetna", "UnitedHealthcare", "Medicare", "Cigna", "Anthem BCBS", "Humana"]
ALLERGY_OPTIONS = [["NKDA"], ["Penicillin"], ["Sulfa drugs"], ["NKDA"], ["Latex"]]
MEDICATION_SETS = [
    ["Metoprolol 25mg BID", "Atorvastatin 40mg QD"],
    ["Lisinopril 10mg QD", "Aspirin 81mg QD"],
    ["Amlodipine 5mg QD", "Clopidogrel 75mg QD"],
]
DIAGNOSTIC_ORDER_TYPES = ["ECG", "Echocardiogram", "Stress test", "Holter monitor", "Coronary angiography"]
DIAGNOSTIC_STATUS_ORDER = ["ordered", "scheduled", "completed"]
CLAIM_STATUS_ORDER = ["pending", "submitted", "approved"]
CPT_BY_ORDER_TYPE = {
    "ECG": ("93000", "ECG w/ interpretation"),
    "Echocardiogram": ("93306", "Echocardiogram, complete"),
    "Stress test": ("93017", "Cardiovascular stress test"),
    "Holter monitor": ("93224", "Holter monitor, 24-48hr"),
    "Coronary angiography": ("93454", "Coronary angiography"),
}


def _seeded(patient_id: str) -> random.Random:
    """Deterministic per-patient PRNG — the same patient_id always seeds the
    same demographic profile within a running process, without needing a
    persisted store for values that don't change record to record."""
    h = int(hashlib.sha256(patient_id.encode()).hexdigest(), 16)
    return random.Random(h)


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: int
    sex: str
    room: str
    cardiologist: str
    nurse: str
    allergies: list[str]
    medications: list[str]
    payer: str
    bp: str = ""
    spo2: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Clinical state — updated live by PatientRegistryAgent
    vitals: dict[str, Any] = field(default_factory=dict)
    vitals_history: list[dict[str, Any]] = field(default_factory=list)  # [{timestamp, hr, qtc, hrv}, ...] — the raw material for longitudinal trending
    quality_score: float = 0.0
    prediction: Optional[dict[str, Any]] = None
    diagnostic_finding: Optional[dict[str, Any]] = None
    automation: Optional[dict[str, Any]] = None

    # Longitudinal trend + multi-modal fusion, for the patient chart
    longitudinal: Optional[dict[str, Any]] = None
    fusion: Optional[dict[str, Any]] = None

    # Diagnostics/treatment — lightly simulated, evolves via API
    diagnostic_order_type: str = "ECG"
    diagnostic_order_status: str = "ordered"
    treatment_plan: str = ""

    # Real clinician-initiated requests — distinct from diagnostic_order_type
    # above, which is just decorative status shown in the billing view. This
    # is the actual "a cardiologist asked for this" trigger that runs the
    # real pipeline, tracked with its result.
    orders: list[dict[str, Any]] = field(default_factory=list)

    # Billing
    claim_status: str = "pending"
    est_reimbursement: int = 0

    # Nursing tasks
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_detail(self) -> dict[str, Any]:
        prediction = self.prediction or {}
        finding = self.diagnostic_finding or {}
        icd10_codes = finding.get("icd10_codes", [])
        cpt_code, cpt_desc = CPT_BY_ORDER_TYPE.get(self.diagnostic_order_type, ("99214", "Office visit, established patient"))
        return {
            "mrn": self.patient_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "room": self.room,
            "cardiologist": self.cardiologist,
            "nurse": self.nurse,
            "diagnosis": finding.get("diagnosis", "No significant finding"),
            "mace_score": prediction.get("score", 0.0),
            "risk_tier": prediction.get("risk_tier", "low"),
            "last_visit": time.strftime("%Y-%m-%d", time.gmtime(self.last_updated)),
            "vitals": {
                "hr": self.vitals.get("hr"), "bp": self.bp, "spo2": self.spo2,
                "qtc": self.vitals.get("qtc"), "hrv": self.vitals.get("hrv"),
            },
            "allergies": self.allergies,
            "medications": self.medications,
            "diagnostic_order": {"type": self.diagnostic_order_type, "status": self.diagnostic_order_status},
            "treatment_plan": self.treatment_plan or "Routine monitoring — no active treatment plan.",
            "payer": self.payer,
            "claim_status": self.claim_status,
            "icd10": {"code": icd10_codes[0], "desc": GENERIC_ABNORMAL_DESC} if icd10_codes else None,
            "cpt": {"code": cpt_code, "desc": cpt_desc},
            "est_reimbursement": self.est_reimbursement,
            "coded": bool(icd10_codes),
            "tasks": self.tasks,
            "automation": self.automation,
            "quality_score": self.quality_score,
            "orders": self.orders,
            "vitals_history_count": len(self.vitals_history),
            "longitudinal": self.longitudinal,
            "fusion": self.fusion,
        }

    def get_vitals_history(self) -> list[dict[str, Any]]:
        return self.vitals_history


class PatientRegistry:
    def __init__(self):
        self.patients: dict[str, PatientRecord] = {}

    def _seed_new(self, patient_id: str) -> PatientRecord:
        rnd = _seeded(patient_id)
        idx = rnd.randint(0, 999)
        record = PatientRecord(
            patient_id=patient_id,
            name=f"Demo Patient {patient_id}",
            age=rnd.randint(40, 85),
            sex=rnd.choice(["F", "M"]),
            room=f"{3 + idx % 4}{chr(65 + idx % 3)}-{10 + idx % 20}",
            cardiologist=rnd.choice(CARDIOLOGISTS),
            nurse=rnd.choice(NURSES),
            allergies=rnd.choice(ALLERGY_OPTIONS),
            medications=rnd.choice(MEDICATION_SETS),
            payer=rnd.choice(PAYERS),
            bp=f"{105 + rnd.randint(0, 35)}/{65 + rnd.randint(0, 20)}",
            spo2=92 + rnd.randint(0, 7),
            diagnostic_order_type=rnd.choice(DIAGNOSTIC_ORDER_TYPES),
            diagnostic_order_status=rnd.choice(DIAGNOSTIC_STATUS_ORDER),
            est_reimbursement=180 + rnd.randint(0, 2400),
        )
        record.tasks = [
            {"id": f"{patient_id}-t1", "label": f"Vitals check — {record.name}", "meta": "Q4H", "done": False},
            {"id": f"{patient_id}-t2", "label": f"Medication administration — {record.name}", "meta": "per MAR", "done": False},
            {"id": f"{patient_id}-t3", "label": f"Escalate if MACE \u2265 0.6 — {record.name}", "meta": "monitoring", "done": False},
        ]
        self.patients[patient_id] = record
        return record

    def get_or_create(self, patient_id: str) -> PatientRecord:
        return self.patients.get(patient_id) or self._seed_new(patient_id)

    def intake_vitals(self, patient_id: str, raw_record: dict[str, Any]) -> PatientRecord:
        """
        Early hop: resolves/creates the patient and appends this record's
        vitals to history — called BEFORE diagnosis/automation run, so
        LongitudinalTrendEngine can compute a trend that INCLUDES the
        current reading, and FusionAgent can combine that trend with the
        ECG score before DiagnosticAgent makes its decision. This used to
        happen only at the very end of the pipeline (in update_clinical,
        now split into this + finalize_clinical below), which meant the
        fused score was computed too late to ever influence the diagnosis
        that had already been made from the raw ECG score alone.
        """
        record = self.get_or_create(patient_id)
        if "heart_rate_bpm" in raw_record:
            record.vitals["hr"] = raw_record["heart_rate_bpm"]
        if "qt_interval_ms" in raw_record:
            record.vitals["qtc"] = raw_record["qt_interval_ms"]
        if "hrv_sdnn_ms" in raw_record:
            record.vitals["hrv"] = raw_record["hrv_sdnn_ms"]

        # Append to history whenever this record actually carries ECG-style
        # vitals — this is the raw material LongitudinalTrendEngine needs.
        # A claims or bare imaging record (no hr/qtc/hrv) shouldn't add a
        # noisy null-filled entry to the trend history.
        if any(k in raw_record for k in ("heart_rate_bpm", "qt_interval_ms", "hrv_sdnn_ms")):
            record.vitals_history.append({
                "timestamp": time.time(),
                "hr": raw_record.get("heart_rate_bpm"),
                "qtc": raw_record.get("qt_interval_ms"),
                "hrv": raw_record.get("hrv_sdnn_ms"),
            })
            if len(record.vitals_history) > 200:  # cap memory growth
                record.vitals_history = record.vitals_history[-200:]

        record.last_updated = time.time()
        return record

    def finalize_clinical(
        self, patient_id: str, quality_score: float, prediction: Optional[dict[str, Any]],
        diagnostic_finding: Optional[dict[str, Any]], automation: Optional[dict[str, Any]],
        longitudinal: Optional[dict[str, Any]] = None, fusion: Optional[dict[str, Any]] = None,
    ) -> PatientRecord:
        """
        Late hop: saves everything the pipeline computed for this record —
        quality, prediction, diagnostic finding, automation tier, and now
        also the longitudinal/fusion assessment that was computed BEFORE
        diagnosis this time (see intake_vitals above), not after. Does NOT
        touch vitals_history — that already happened in intake_vitals, and
        calling it again here would double-append the same reading.
        """
        record = self.get_or_create(patient_id)
        record.quality_score = quality_score
        record.prediction = prediction
        record.diagnostic_finding = diagnostic_finding
        record.automation = automation
        if longitudinal is not None:
            record.longitudinal = longitudinal
        if fusion is not None:
            record.fusion = fusion
        record.last_updated = time.time()
        if diagnostic_finding and diagnostic_finding.get("specialist_review_required"):
            record.treatment_plan = (
                f"Guideline-directed evaluation for: {diagnostic_finding['diagnosis']}. "
                f"Cardiology follow-up recommended."
            )
        return record

    def list_all(self) -> list[dict[str, Any]]:
        return [p.to_detail() for p in sorted(self.patients.values(), key=lambda x: x.last_updated, reverse=True)]

    def get_detail(self, patient_id: str) -> Optional[dict[str, Any]]:
        p = self.patients.get(patient_id)
        return p.to_detail() if p else None

    def get_vitals_history(self, patient_id: str) -> Optional[list[dict[str, Any]]]:
        p = self.patients.get(patient_id)
        return p.get_vitals_history() if p else None

    def toggle_task(self, patient_id: str, task_id: str) -> Optional[dict[str, Any]]:
        p = self.patients.get(patient_id)
        if not p:
            return None
        for t in p.tasks:
            if t["id"] == task_id:
                t["done"] = not t["done"]
                break
        return p.to_detail()

    def advance_claim(self, patient_id: str) -> Optional[dict[str, Any]]:
        p = self.patients.get(patient_id)
        if not p:
            return None
        if p.claim_status in CLAIM_STATUS_ORDER:
            idx = CLAIM_STATUS_ORDER.index(p.claim_status)
            if idx < len(CLAIM_STATUS_ORDER) - 1:
                p.claim_status = CLAIM_STATUS_ORDER[idx + 1]
        return p.to_detail()

    def advance_diagnostic_order(self, patient_id: str) -> Optional[dict[str, Any]]:
        p = self.patients.get(patient_id)
        if not p:
            return None
        if p.diagnostic_order_status in DIAGNOSTIC_STATUS_ORDER:
            idx = DIAGNOSTIC_STATUS_ORDER.index(p.diagnostic_order_status)
            if idx < len(DIAGNOSTIC_STATUS_ORDER) - 1:
                p.diagnostic_order_status = DIAGNOSTIC_STATUS_ORDER[idx + 1]
        return p.to_detail()

    def add_order(self, patient_id: str, order_type: str, ordered_by: str, status: str, result_summary: str) -> dict[str, Any]:
        """
        Records a real clinician-initiated request — the thing that was
        missing before: not data arriving, but a person asking for analysis.
        The actual pipeline dispatch (for order types the engine can fulfill)
        happens in the API layer, which then calls this to log the result;
        keeping that orchestration out of the registry avoids a circular
        dependency between PatientRegistry and the orchestrator that owns it.
        """
        p = self.get_or_create(patient_id)
        order = {
            "order_id": f"ORD-{len(p.orders) + 1:03d}-{patient_id}",
            "order_type": order_type,
            "ordered_by": ordered_by,
            "status": status,
            "result_summary": result_summary,
            "created_at": time.time(),
        }
        p.orders.insert(0, order)  # newest first
        return order
