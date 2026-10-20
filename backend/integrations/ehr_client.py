"""
Live EHR connection over FHIR R4 (read a patient's latest vitals; write a MACE result back).

Configured only by environment variables, so a deployment with no EHR connection behaves exactly as before:
    EHR_FHIR_BASE_URL   e.g. https://<hospital>/fhir/R4
    EHR_FHIR_TOKEN      bearer token (SMART on FHIR / system-to-system). Never logged or returned.

NOT TESTED AGAINST A REAL EHR. It follows the FHIR R4 REST spec and is tested against a mock server only. Real Epic or
Cerner connections add app registration, scopes, token refresh and field mapping that must be validated with the
hospital's integration team before clinical use.

What a standard FHIR vitals feed carries: heart rate, QTc, SpO2 and demographics. It does NOT carry the raw 10-second
ECG waveform; that comes from MUSE or Mayo's ECG tables and is attached to the chart separately.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Optional

import httpx

LOINC_CODES = {"heart_rate_bpm": "8867-4", "qt_interval_ms": "8634-8", "spo2_pct": "59408-5"}


class EHRError(Exception):
    pass


class EHRClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 10.0, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Accept": "application/fhir+json", **({"Authorization": f"Bearer {token}"} if token else {})}
        self._client = httpx.Client(timeout=timeout, transport=transport)

    @classmethod
    def from_env(cls) -> Optional["EHRClient"]:
        base = os.environ.get("EHR_FHIR_BASE_URL", "").strip()
        return cls(base, os.environ.get("EHR_FHIR_TOKEN") or None) if base else None

    @property
    def host(self) -> str:
        return httpx.URL(self.base_url).host

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        try:
            r = self._client.get(f"{self.base_url}/{path}", params=params, headers=self._headers)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise EHRError(f"EHR returned HTTP {exc.response.status_code} for {path.split('?')[0]}") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise EHRError(f"Could not reach the EHR: {type(exc).__name__}") from None

    def fetch_patient_snapshot(self, patient_id: str) -> dict[str, Any]:
        """-> {age, sex, vitals: {heart_rate_bpm, qt_interval_ms, spo2_pct (those found)}, observed_at}"""
        pat = self._get(f"Patient/{patient_id}")
        age = None
        if pat.get("birthDate"):
            try:
                b = date.fromisoformat(pat["birthDate"][:10]); t = date.today()
                age = t.year - b.year - ((t.month, t.day) < (b.month, b.day))
            except ValueError:
                pass
        sex = {"male": "M", "female": "F"}.get((pat.get("gender") or "").lower())
        vitals, newest = {}, None
        for key, code in LOINC_CODES.items():
            bundle = self._get("Observation", {"patient": patient_id, "code": f"http://loinc.org|{code}", "_sort": "-date", "_count": "1"})
            for e in bundle.get("entry", []):
                res = e.get("resource", {})
                v = (res.get("valueQuantity") or {}).get("value")
                if v is not None:
                    vitals[key] = float(v)
                    when = res.get("effectiveDateTime")
                    newest = max(newest, when) if newest and when else (newest or when)
                    break
        return {"age": age, "sex": sex, "vitals": vitals, "observed_at": newest}

    def post_resource(self, resource: dict) -> dict:
        try:
            r = self._client.post(f"{self.base_url}/{resource['resourceType']}", json=resource, headers={**self._headers, "Content-Type": "application/fhir+json"})
            r.raise_for_status()
            return {"http_status": r.status_code, "location": r.headers.get("Location")}
        except httpx.HTTPStatusError as exc:
            raise EHRError(f"EHR rejected the write with HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise EHRError(f"Could not reach the EHR: {type(exc).__name__}") from None
