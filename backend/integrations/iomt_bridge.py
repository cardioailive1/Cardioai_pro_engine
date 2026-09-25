"""
CardioAI Pro — IoMT Backend Bridge
=======================================
Connects the external, separately-deployed "IoMT CardioAI Backend"
(https://cardioailiverpm.com — a real, live, different codebase with its
own auth, device registry, BLE pairing, implant registration, and vendor
gateway) to THIS system's actual clinical pipeline (quality scoring,
inference, diagnosis, automation tier, longitudinal trend, fusion).

WHY THIS MODULE EXISTS: the IoMT backend's documented endpoints are a
front door for device/vendor data (POST /vendor-gateway/ingest) and a
read-only surface for alerts/reports (GET /alerts, GET /reports) — but
nothing in its actual clinical intelligence. That's what this codebase
already has, tested, across every prior module in this project. This
bridge is the seam: the IoMT backend's vendor-gateway forwards a reading
here; this module translates it into the same modality/record shape
`/api/ingest` already accepts and runs it through the same real pipeline.

WHAT'S VERIFIED AND WHAT'S ASSUMED, STATED PLAINLY:
- Everything on THIS side (translation logic, the orchestrator dispatch,
  the response shape) is real, tested code — see the test run referenced
  in this module's accompanying test script.
- The INBOUND payload shape below (IoMTReading / IoMTIngestRequest) is an
  INFERRED schema, not a confirmed one — built from the endpoint's name
  and general BLE/wearable-gateway conventions (device_id, a reading
  type, a value, a timestamp), since the live system's actual OpenAPI
  spec / request schema was not available to design against directly.
  Confirm the real payload shape against that system's documentation (or
  whoever maintains it) before wiring a live connection — a shape
  mismatch here would fail loudly (a 422 from this endpoint's Pydantic
  validation), not silently, but it's still worth getting right before
  connecting real traffic.
- THE OUTBOUND DIRECTION IS AN OPEN QUESTION, NOT BUILT: the IoMT
  backend's documented endpoints only show GET /alerts and GET /reports
  — no POST. There is no documented way for this system to push a
  computed risk score, diagnosis, or automation-tier decision back INTO
  the IoMT backend for its consumer app or clinical dashboard to display.
  Either that system has an undocumented write endpoint for computed
  clinical output, or it's designed to pull from this system instead of
  receiving pushes, or that direction simply doesn't exist yet. This is
  a real question to resolve with whoever controls that deployment —
  this module does not guess at an endpoint that may not exist.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from orchestrator.orchestrator import orchestrator
from integrations.iomt_client import IoMTBackendClient

router = APIRouter(prefix="/iomt-bridge", tags=["iomt-bridge"])

# Shared-secret check for this specific bridge endpoint — the rest of
# this system still has no auth layer at all (a known, documented gap
# everywhere else in this project), but a new inbound connection point
# from an external, internet-reachable system is exactly the place to
# not extend that gap further. Set BRIDGE_API_KEY as a real secret in
# deployment; an empty/unset value here means the check is a no-op,
# which is intentional for local testing but should never be true in
# a real deployment reachable from cardioailiverpm.com.
BRIDGE_API_KEY = os.environ.get("IOMT_BRIDGE_API_KEY", "")


class IoMTReading(BaseModel):
    """
    One reading, as inferred from the vendor-gateway's naming convention.
    reading_type maps to CardioAI Pro's existing vitals vocabulary — see
    READING_TYPE_MAP below for the exact mapping and which types are
    currently supported.
    """
    reading_type: str
    value: float
    unit: Optional[str] = None
    recorded_at: Optional[str] = None  # ISO 8601; informational only — dispatch uses receipt time, see module note below


class IoMTIngestRequest(BaseModel):
    device_id: str
    patient_id: str  # assumed to be the same identifier space as CardioAI Pro's patient_id / MRN — confirm this with whoever maps IoMT consumer accounts to clinical patient records, since a consumer-app user_id and a hospital MRN are not automatically the same thing
    vendor: Optional[str] = None
    readings: list[IoMTReading]


# Maps the inferred IoMT reading_type vocabulary to the exact record keys
# CardioAI Pro's existing pipeline already understands — both the core
# ECG three and the expanded nurse-charted vitals set (bp/spo2/resp/temp/
# weight) built earlier in this project. Extend this map, don't invent a
# new ingestion path, if the real vendor-gateway uses different type
# strings than guessed here.
READING_TYPE_MAP: dict[str, str] = {
    "heart_rate": "heart_rate_bpm", "hr": "heart_rate_bpm",
    "qtc": "qt_interval_ms", "qt_interval": "qt_interval_ms",
    "hrv": "hrv_sdnn_ms", "hrv_sdnn": "hrv_sdnn_ms",
    "spo2": "spo2_pct", "blood_oxygen": "spo2_pct",
    "bp_systolic": "bp_systolic", "systolic": "bp_systolic",
    "bp_diastolic": "bp_diastolic", "diastolic": "bp_diastolic",
    "resp_rate": "resp_rate_bpm", "respiratory_rate": "resp_rate_bpm",
    "temp": "temp_f", "temperature": "temp_f",
    "weight": "weight_lb",
    "steps": "steps",
}

# A record carrying any of these keys is real ECG-pipeline input (real
# inference exists for this modality); a record with only the wearable-
# style fields routes to "wearable" instead — matches how
# ingestion/quality.py's SCHEMAS already distinguish the two.
ECG_SIGNAL_KEYS = {"heart_rate_bpm", "qt_interval_ms", "hrv_sdnn_ms"}


def _check_bridge_auth(x_bridge_api_key: Optional[str]) -> None:
    if BRIDGE_API_KEY and x_bridge_api_key != BRIDGE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Bridge-Api-Key.")


def translate_iomt_payload(payload: IoMTIngestRequest) -> tuple[str, dict[str, Any]]:
    """
    Returns (modality, record) in exactly the shape /api/ingest already
    accepts. Unrecognized reading_types are dropped with a note in the
    returned record under "_unmapped_readings" rather than silently
    discarded — so a real payload using different type strings than
    guessed here is visible in the response, not invisible.
    """
    record: dict[str, Any] = {"patient_id": payload.patient_id, "device_id": payload.device_id}
    unmapped: list[str] = []

    for reading in payload.readings:
        mapped_key = READING_TYPE_MAP.get(reading.reading_type.lower())
        if mapped_key is None:
            unmapped.append(reading.reading_type)
            continue
        record[mapped_key] = reading.value

    if unmapped:
        record["_unmapped_readings"] = unmapped

    modality = "ecg" if any(k in record for k in ECG_SIGNAL_KEYS) else "wearable"
    return modality, record


@router.post("/ingest")
async def iomt_bridge_ingest(payload: IoMTIngestRequest, x_bridge_api_key: Optional[str] = Header(None)):
    """
    Receives a reading forwarded from the IoMT backend's vendor-gateway,
    translates it, and runs it through the exact same orchestrator
    pipeline /api/ingest uses — no separate, parallel logic to keep in
    sync. See this module's docstring for what's verified vs. assumed
    about the inbound shape, and why the outbound direction (pushing
    results back to the IoMT backend) isn't built here.
    """
    _check_bridge_auth(x_bridge_api_key)

    if not payload.readings:
        raise HTTPException(status_code=400, detail="No readings in payload.")

    modality, record = translate_iomt_payload(payload)

    try:
        result = await orchestrator.dispatch(modality, record, source=f"iomt-bridge:{payload.vendor or payload.device_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response: dict[str, Any] = {
        "received_at": time.time(),
        "translated_modality": modality,
        "translated_record": record,
        "pipeline_result": result,
    }
    if "_unmapped_readings" in record:
        response["warning"] = (
            f"Reading type(s) {record['_unmapped_readings']} were not recognized and were dropped — "
            f"see READING_TYPE_MAP in integrations/iomt_bridge.py to add support for them."
        )

    # Best-effort push-back — the proposed results endpoint on the IoMT
    # backend doesn't exist yet (see integrations/iomt_client.py's
    # docstring), so this fails today, on purpose, and that failure does
    # NOT fail the whole request: the inbound processing above already
    # succeeded and is the valuable part. Once that endpoint is built,
    # this starts working with no further change needed on this side.
    inner = result.get("result", {})
    try:
        push_result = await IoMTBackendClient().push_clinical_result(
            patient_id=payload.patient_id, prediction=inner.get("prediction"),
            diagnostic_finding=inner.get("diagnostic_finding"), automation=inner.get("automation"),
        )
        response["pushed_to_iomt_backend"] = push_result
    except Exception as exc:
        response["push_back_status"] = f"not pushed — {exc}"

    return response
