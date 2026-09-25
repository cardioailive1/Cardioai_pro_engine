"""
CardioAI Pro — IoMT Backend Client (outbound)
===================================================
The other half of integrations/iomt_bridge.py. That module RECEIVES
pushes from the IoMT backend; this module lets CardioAI Pro call OUT to
it — both to pull data (GET) and to push computed clinical results back
(POST).

WHAT'S REAL HERE AND WHAT ISN'T:
- The GET methods (get_devices, get_alerts, get_reports) call the IoMT
  backend's actual documented endpoints. The request/response handling
  is real, tested code — but tested against a LOCAL MOCK standing in for
  the real system (see training/mock_iomt_backend_test.py), because this
  project has no credentials to the live deployment. Response field names
  below are best-effort guesses from the endpoint names alone; confirm
  against the real response shape before depending on specific fields.
- The POST method (push_clinical_result) calls an endpoint —
  `/clinical/cardioai-pro/results` — THAT DOES NOT EXIST YET on the live
  IoMT backend. This is a proposed spec, not a confirmed integration.
  Calling it against the real system today will 404. See this project's
  README for the exact request/response contract to build on that side,
  and for why guessing at an existing endpoint (rather than proposing a
  new, clearly-named one) would have been worse — reusing e.g.
  /vendor-gateway/ingest for computed clinical output would misrepresent
  an AI-derived risk score as raw vendor device data to whatever system
  consumes it downstream.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx

IOMT_BACKEND_URL = os.environ.get("IOMT_BACKEND_URL", "https://cardioailiverpm.com")
IOMT_BACKEND_API_KEY = os.environ.get("IOMT_BACKEND_API_KEY", "")


class IoMTBackendClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 10.0):
        self.base_url = (base_url or IOMT_BACKEND_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else IOMT_BACKEND_API_KEY
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Vendor-Api-Key"] = self.api_key  # matches the IoMT backend's own documented auth header name for its vendor-facing endpoints
        return headers

    async def get_devices(self, patient_id: Optional[str] = None) -> dict[str, Any]:
        """Pulls the registered device list — GET /devices, auth required."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {"patient_id": patient_id} if patient_id else None
            resp = await client.get(f"{self.base_url}/devices", headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_alerts(self) -> dict[str, Any]:
        """Pulls active alerts — GET /alerts, auth required."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/alerts", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def get_reports(self) -> dict[str, Any]:
        """Pulls clinical reports — GET /reports, auth required."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/reports", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def push_clinical_result(
        self, patient_id: str, prediction: Optional[dict[str, Any]],
        diagnostic_finding: Optional[dict[str, Any]], automation: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        PROPOSED endpoint, not yet built on the IoMT backend — see module
        docstring. Pushes this engine's computed output back so the
        consumer app / clinical dashboard on that system can surface it.
        Raises httpx.HTTPStatusError (404 today) until that endpoint exists.
        """
        payload = {
            "patient_id": patient_id,
            "source_system": "cardioai-pro",
            "risk_score": prediction.get("score") if prediction else None,
            "risk_tier": prediction.get("risk_tier") if prediction else None,
            "diagnosis": diagnostic_finding.get("diagnosis") if diagnostic_finding else None,
            "icd10_codes": diagnostic_finding.get("icd10_codes") if diagnostic_finding else [],
            "automation_tier": automation.get("tier") if automation else None,
            "requires_signoff": automation.get("requires_human_signoff") if automation else None,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/clinical/cardioai-pro/results", headers=self._headers(), json=payload,
            )
            resp.raise_for_status()
            return resp.json()
