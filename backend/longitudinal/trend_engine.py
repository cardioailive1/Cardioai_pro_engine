"""
CardioAI Pro — Longitudinal Trend Engine
=============================================
Every other risk signal in this engine looks at ONE observation in
isolation — a single ECG reading, scored and gone. That's a structural
reason the "30-90 days early" claim couldn't be true: genuine early
detection needs to see a developing TREND, not just an absolute value.
A resting heart rate of 95 might be normal for one patient and a real
change for another — what matters is the deviation from THEIR OWN
baseline, not a population cutoff.

This module computes real statistics — baseline, current deviation,
linear trend slope over time, volatility — from a patient's actual
accumulated reading history (orchestrator/patient_registry.py's
vitals_history, populated as records flow through the pipeline). It does
NOT predict anything itself; it produces features a diagnostic/fusion
step can use, the same way inference/models.py produces a risk score from
a single reading. Combining the two is what inference/multimodal_fusion.py
does.

Needs a minimum number of observations to say anything meaningful — with
one or two readings there's no trend to compute, only noise, and this
module says so explicitly rather than returning a confident-looking
number from insufficient data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

MIN_OBSERVATIONS_FOR_TREND = 4
MIN_TIME_SPAN_DAYS = 0.25  # ~6 hours — below this, a "per day" slope is numerically unstable (dividing by a near-zero time span), not just imprecise
BASELINE_WINDOW_FRACTION = 0.34  # earliest third of history defines "baseline"


@dataclass
class FeatureTrend:
    feature: str
    n_observations: int
    baseline_mean: Optional[float] = None
    current_value: Optional[float] = None
    deviation_from_baseline: Optional[float] = None       # current - baseline, signed
    deviation_pct: Optional[float] = None                   # as a % of baseline
    slope_per_day: Optional[float] = None                    # linear trend, units/day
    volatility: Optional[float] = None                        # std dev across the window
    sufficient_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature, "n_observations": self.n_observations,
            "baseline_mean": self.baseline_mean, "current_value": self.current_value,
            "deviation_from_baseline": self.deviation_from_baseline, "deviation_pct": self.deviation_pct,
            "slope_per_day": self.slope_per_day, "volatility": self.volatility,
            "sufficient_data": self.sufficient_data,
        }


@dataclass
class LongitudinalAssessment:
    patient_id: str
    n_observations: int
    days_of_history: float
    sufficient_data: bool
    trends: dict[str, FeatureTrend] = field(default_factory=dict)
    trend_risk_contribution: float = 0.0    # 0-1, how concerning the TREND itself looks, independent of current value
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id, "n_observations": self.n_observations,
            "days_of_history": round(self.days_of_history, 2), "sufficient_data": self.sufficient_data,
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "trend_risk_contribution": round(self.trend_risk_contribution, 4),
            "narrative": self.narrative,
        }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least squares slope — the trend direction/rate, in y-units per x-unit (day)."""
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


class LongitudinalTrendEngine:
    FEATURES = ["hr", "qtc", "hrv"]
    # Direction that counts as "worsening" for each feature — used to build
    # the single trend_risk_contribution score. Rising HR is concerning;
    # falling HRV is concerning; QTc has no single "good direction" bias
    # built in here since both prolongation and (rarer) shortening can
    # matter, so it contributes via volatility rather than slope direction.
    WORSENING_DIRECTION = {"hr": +1, "hrv": -1}

    def assess(self, patient_id: str, history: list[dict[str, Any]]) -> LongitudinalAssessment:
        if not history:
            return LongitudinalAssessment(
                patient_id=patient_id, n_observations=0, days_of_history=0.0, sufficient_data=False,
                narrative="No history yet — this is the first reading on record.",
            )

        history = sorted(history, key=lambda h: h["timestamp"])
        n = len(history)
        t0 = history[0]["timestamp"]
        times_days = np.array([(h["timestamp"] - t0) / 86400.0 for h in history])
        days_of_history = float(times_days[-1] - times_days[0])
        sufficient = n >= MIN_OBSERVATIONS_FOR_TREND

        trends: dict[str, FeatureTrend] = {}
        worsening_signals: list[float] = []

        for feat in self.FEATURES:
            values = np.array([h.get(feat) for h in history], dtype=float)
            valid = ~np.isnan(values)
            if valid.sum() < 2:
                trends[feat] = FeatureTrend(feature=feat, n_observations=int(valid.sum()), sufficient_data=False)
                continue

            v = values[valid]
            t = times_days[valid]
            baseline_n = max(1, int(len(v) * BASELINE_WINDOW_FRACTION))
            baseline_mean = float(v[:baseline_n].mean())
            current_value = float(v[-1])
            deviation = current_value - baseline_mean
            deviation_pct = (deviation / baseline_mean * 100) if baseline_mean else None
            time_span_days = float(t.max() - t.min()) if len(t) else 0.0
            # Slope needs BOTH enough points and enough time spread — with
            # readings seconds apart (e.g. rapid API calls, or continuous
            # telemetry), the time-span denominator is near-zero and a naive
            # slope calculation blows up into a nonsense number (this was a
            # real bug: readings ~1s apart produced a slope of ~62 million
            # units/day). Below the minimum span, slope is honestly None,
            # not a number that LOOKS precise but isn't meaningful.
            slope = (
                _linear_slope(t, v)
                if valid.sum() >= MIN_OBSERVATIONS_FOR_TREND and time_span_days >= MIN_TIME_SPAN_DAYS
                else None
            )
            volatility = float(v.std()) if len(v) >= 2 else None

            ft = FeatureTrend(
                feature=feat, n_observations=int(valid.sum()), baseline_mean=round(baseline_mean, 2),
                current_value=round(current_value, 2), deviation_from_baseline=round(deviation, 2),
                deviation_pct=round(deviation_pct, 2) if deviation_pct is not None else None,
                slope_per_day=round(slope, 4) if slope is not None else None,
                volatility=round(volatility, 2) if volatility is not None else None,
                sufficient_data=bool(valid.sum() >= MIN_OBSERVATIONS_FOR_TREND and time_span_days >= MIN_TIME_SPAN_DAYS),
            )
            trends[feat] = ft

            # Contribute to the combined trend-risk score only when there's
            # both enough data AND a defined "worsening direction" for this feature.
            if ft.sufficient_data and feat in self.WORSENING_DIRECTION and slope is not None:
                direction = self.WORSENING_DIRECTION[feat]
                # Normalize slope by a feature-appropriate scale so hr/hrv slopes are comparable.
                # Calibrated so a sustained ~1-1.5 units/day drift (e.g. HR climbing
                # from 70 to 100+ over a month, or HRV roughly halving) registers
                # as clearly concerning rather than needing an extreme slope to trip.
                scale = {"hr": 2.0, "hrv": 2.0}.get(feat, 1.0)
                signal = max(0.0, direction * slope / scale)
                worsening_signals.append(min(signal, 1.0))

        trend_risk = float(np.mean(worsening_signals)) if worsening_signals else 0.0

        narrative = self._narrative(n, days_of_history, sufficient, trends, trend_risk)

        return LongitudinalAssessment(
            patient_id=patient_id, n_observations=n, days_of_history=days_of_history,
            sufficient_data=sufficient, trends=trends, trend_risk_contribution=round(trend_risk, 4),
            narrative=narrative,
        )

    def _narrative(self, n, days, sufficient, trends, trend_risk) -> str:
        if not sufficient:
            return f"Only {n} observation(s) on record — need at least {MIN_OBSERVATIONS_FOR_TREND} to compute a real trend, not just noise."
        if days < MIN_TIME_SPAN_DAYS:
            return (
                f"{n} readings on record, but they span only {days*24:.1f} hours — need at least "
                f"{MIN_TIME_SPAN_DAYS*24:.0f} hours of spread before a per-day trend slope is numerically "
                f"reliable, not just enough readings. Baseline/current values are available; trend direction is not yet."
            )
        parts = []
        hr = trends.get("hr")
        hrv = trends.get("hrv")
        if hr and hr.sufficient_data and hr.slope_per_day is not None:
            direction = "rising" if hr.slope_per_day > 0.5 else "falling" if hr.slope_per_day < -0.5 else "stable"
            parts.append(f"heart rate {direction} ({hr.slope_per_day:+.2f} bpm/day)")
        if hrv and hrv.sufficient_data and hrv.slope_per_day is not None:
            direction = "declining" if hrv.slope_per_day < -0.3 else "improving" if hrv.slope_per_day > 0.3 else "stable"
            parts.append(f"HRV {direction} ({hrv.slope_per_day:+.2f} ms/day)")
        trend_desc = "; ".join(parts) if parts else "no strong directional trend detected"
        risk_desc = "concerning trend pattern" if trend_risk > 0.5 else "mild trend signal" if trend_risk > 0.15 else "no concerning trend"
        return f"Over {days:.1f} days across {n} readings: {trend_desc}. Assessment: {risk_desc}."
