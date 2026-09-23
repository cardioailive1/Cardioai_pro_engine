"""
CardioAI Pro — HL7 v2.x Integration
=======================================
Many hospital systems still speak HL7 v2 (pipe-delimited) alongside or
instead of FHIR, especially for ADT (admit/discharge/transfer) and ORU
(observation result) feeds over an interface engine (Mirth, Rhapsody, etc.).

This module builds and parses the two message types CardioScan Pro cares
about:
  - ADT^A01 (admit)      -> triggers a new patient context
  - ORU^R01 (result)     -> carries a MACE risk score back into the EHR

A production deployment listens on a persistent MLLP socket (port 2575 is
conventional) rather than HTTP; `build_oru_r01` / `parse_hl7_message` are
the pieces that plug into that listener.
"""
from __future__ import annotations

import time
from typing import Any

SEG_SEP = "\r"
FIELD_SEP = "|"


def _hl7_timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())


def build_adt_a01(patient_id: str, patient_name: str, admit_location: str) -> str:
    msh = f"MSH|^~\\&|CARDIOAI|CARDIOSCAN|EHR|HOSPITAL|{_hl7_timestamp()}||ADT^A01|{patient_id}-ADT|P|2.5"
    pid = f"PID|1||{patient_id}||{patient_name}"
    pv1 = f"PV1|1|I|{admit_location}"
    return SEG_SEP.join([msh, pid, pv1])


def build_oru_r01(patient_id: str, risk_score: float, risk_tier: str, window_days: int) -> str:
    """Carries a CardioScan Pro MACE risk result back into the EHR as an ORU."""
    msh = f"MSH|^~\\&|CARDIOAI|CARDIOSCAN|EHR|HOSPITAL|{_hl7_timestamp()}||ORU^R01|{patient_id}-ORU|P|2.5"
    pid = f"PID|1||{patient_id}"
    obr = f"OBR|1||{patient_id}-MACE|MACE^CardioAI MACE Risk Score^CARDIOAI"
    obx = (
        f"OBX|1|NM|MACE-RISK^{window_days}-day MACE risk^CARDIOAI||"
        f"{risk_score:.2f}|score|{_risk_range(risk_tier)}|{risk_tier.upper()}|||F"
    )
    return SEG_SEP.join([msh, pid, obr, obx])


def _risk_range(tier: str) -> str:
    return {"low": "0.00-0.30", "moderate": "0.30-0.60", "high": "0.60-1.00"}.get(tier, "")


def parse_hl7_message(message: str) -> dict[str, Any]:
    """Minimal pipe-delimited parser sufficient for MSH/PID/OBX inspection."""
    segments = [s for s in message.replace("\n", SEG_SEP).split(SEG_SEP) if s]
    parsed: dict[str, Any] = {"segments": []}
    for seg in segments:
        fields = seg.split(FIELD_SEP)
        parsed["segments"].append({"type": fields[0], "fields": fields[1:]})
        if fields[0] == "MSH":
            parsed["message_type"] = fields[8] if len(fields) > 8 else None
            parsed["control_id"] = fields[9] if len(fields) > 9 else None
    return parsed
