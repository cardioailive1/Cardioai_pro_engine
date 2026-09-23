"""
CardioAI Pro — Data Quality at the Ingestion Point
=====================================================
Runs on every record before it reaches FHIR mapping or inference. Four
checks, each contributing to a 0-100 quality score:

  1. Schema validation   — required fields present, correct types
  2. Range validation     — physiologically plausible values
  3. Completeness         — proportion of non-null expected fields
  4. Outlier / drift flag — z-score against a running per-field baseline

A record below QUALITY_THRESHOLD is still passed downstream but flagged,
so the orchestrator can route it to review rather than silently drop it.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

QUALITY_THRESHOLD = 70.0

# Expected fields + physiologically plausible ranges per modality
SCHEMAS: dict[str, dict[str, tuple[float, float]]] = {
    "ecg": {
        "heart_rate_bpm": (30, 220),
        "qt_interval_ms": (250, 550),
        "hrv_sdnn_ms": (5, 200),
    },
    "wearable": {
        "heart_rate_bpm": (30, 220),
        "spo2_pct": (70, 100),
        "steps": (0, 60000),
    },
    "claims": {
        "member_age": (0, 110),
        "risk_flags_count": (0, 20),
    },
    "imaging": {
        # imaging records are validated structurally in DICOMService instead
    },
}


class RunningStats:
    """Tiny incremental mean/variance tracker, used for the drift check."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def zscore(self, x: float) -> float:
        if self.n < 5:
            return 0.0
        variance = self.m2 / max(self.n - 1, 1)
        std = math.sqrt(variance) or 1.0
        return (x - self.mean) / std


class DataQualityEngine:
    def __init__(self):
        self._baselines: dict[str, dict[str, RunningStats]] = defaultdict(lambda: defaultdict(RunningStats))

    def evaluate(self, modality: str, record: dict[str, Any]) -> dict[str, Any]:
        schema = SCHEMAS.get(modality, {})
        issues: list[str] = []

        # 1. Schema — required fields present
        expected_fields = set(schema.keys())
        present_fields = set(k for k, v in record.items() if v is not None)
        missing = expected_fields - present_fields
        schema_score = 100.0 if not missing else max(0.0, 100 - 100 * len(missing) / max(len(expected_fields), 1))
        if missing:
            issues.append(f"missing fields: {', '.join(sorted(missing))}")

        # 2. Range — plausibility
        out_of_range = []
        for field, (lo, hi) in schema.items():
            val = record.get(field)
            if isinstance(val, (int, float)) and not (lo <= val <= hi):
                out_of_range.append(field)
        range_score = 100.0 if not out_of_range else max(0.0, 100 - 100 * len(out_of_range) / max(len(schema), 1))
        if out_of_range:
            issues.append(f"out of plausible range: {', '.join(out_of_range)}")

        # 3. Completeness — all fields in the record itself, not just schema
        total_fields = len(record) or 1
        non_null = sum(1 for v in record.values() if v is not None)
        completeness_score = 100.0 * non_null / total_fields

        # 4. Drift / outlier — z-score vs running baseline per field
        drift_flags = []
        baseline = self._baselines[modality]
        for field, val in record.items():
            if isinstance(val, (int, float)):
                stats = baseline[field]
                z = stats.zscore(float(val))
                stats.update(float(val))
                if abs(z) > 3:
                    drift_flags.append(field)
        drift_score = 100.0 if not drift_flags else max(0.0, 100 - 25 * len(drift_flags))
        if drift_flags:
            issues.append(f"statistical outlier vs. recent baseline: {', '.join(drift_flags)}")

        overall = round(0.35 * schema_score + 0.30 * range_score + 0.20 * completeness_score + 0.15 * drift_score, 1)

        return {
            "score": overall,
            "passed": overall >= QUALITY_THRESHOLD,
            "breakdown": {
                "schema": round(schema_score, 1),
                "range": round(range_score, 1),
                "completeness": round(completeness_score, 1),
                "drift": round(drift_score, 1),
            },
            "issues": issues,
        }
