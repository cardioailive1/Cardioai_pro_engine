"""
CardioAI Pro — Live Data Streaming
=====================================
Simulates the three inbound streams described in the customer use-case
diagram: EHR/ECG feed (hospital), claims feed (payer), wearable feed
(consumer app). In production these generators are replaced 1:1 with:

  - ecg_stream   -> Epic/Cerner FHIR subscription or HL7 ORU feed
  - claims_stream -> payer batch/SFTP or FHIR Bulk Data export
  - wearable_stream -> mobile app event stream (Kafka/SQS/etc.)

The orchestrator.dispatch() call is identical either way — only where the
record comes from changes.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncGenerator


def _jitter(base: float, spread: float) -> float:
    return round(base + random.uniform(-spread, spread), 1)


async def ecg_stream(interval: float = 1.5) -> AsyncGenerator[dict, None]:
    patient_ids = [f"P-{i:04d}" for i in range(1, 40)]
    while True:
        yield {
            "modality": "ecg",
            "source": "Epic FHIR subscription (simulated)",
            "record": {
                "patient_id": random.choice(patient_ids),
                "heart_rate_bpm": _jitter(78, 35),
                "qt_interval_ms": _jitter(400, 60),
                "hrv_sdnn_ms": _jitter(55, 40),
            },
        }
        await asyncio.sleep(interval)


async def wearable_stream(interval: float = 2.0) -> AsyncGenerator[dict, None]:
    subscriber_ids = [f"SUB-{i:04d}" for i in range(1, 200)]
    while True:
        yield {
            "modality": "wearable",
            "source": "iOS/Android app (simulated)",
            "record": {
                "subscriber_id": random.choice(subscriber_ids),
                "heart_rate_bpm": _jitter(72, 30),
                "spo2_pct": _jitter(97, 5),
                "steps": random.randint(0, 500),
            },
        }
        await asyncio.sleep(interval)


async def claims_stream(interval: float = 3.0) -> AsyncGenerator[dict, None]:
    while True:
        yield {
            "modality": "claims",
            "source": "Payer batch feed (simulated)",
            "record": {
                "member_id": f"M-{random.randint(10000, 99999)}",
                "member_age": random.randint(28, 89),
                "risk_flags_count": random.randint(0, 6),
            },
        }
        await asyncio.sleep(interval)


STREAMS = {
    "ecg": ecg_stream,
    "wearable": wearable_stream,
    "claims": claims_stream,
}
