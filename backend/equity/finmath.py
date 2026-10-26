"""Small financial-math helpers shared by the equity engines (no third-party packages)."""
from __future__ import annotations

import math
from datetime import date
from typing import Callable, Sequence


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, r: float, vol: float, q: float = 0.0) -> float:
    """Black-Scholes European call. s = underlying, k = strike."""
    if s <= 0:
        return 0.0
    if k <= 0:
        return s * math.exp(-q * t)
    if t <= 0 or vol <= 0:
        return max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return s * math.exp(-q * t) * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def finnerty_dlom(vol: float, t: float) -> float:
    """Finnerty (2012) average-strike put model for the discount for lack of marketability, no dividends."""
    v2t = vol * vol * t
    inner = v2t + math.log(2 * (math.exp(v2t) - v2t - 1)) - 2 * math.log(math.exp(v2t) - 1)
    v_sqrt_t = math.sqrt(max(inner, 0.0))
    return norm_cdf(v_sqrt_t / 2) - norm_cdf(-v_sqrt_t / 2)


def bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-9, max_iter: int = 200) -> float:
    """Root of an increasing/decreasing f on [lo, hi]. Raises ValueError if the root is not bracketed."""
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("root not bracketed")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) / 2 < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def months_between(a: date, b: date) -> int:
    """Whole months from a to b (b on or after a)."""
    m = (b.year - a.year) * 12 + (b.month - a.month)
    return m - (1 if b.day < a.day else 0)


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) + n, 12)
    last = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
    return date(y, m + 1, min(d.day, last))


def xirr(flows: Sequence[tuple[date, float]]) -> float | None:
    """Annualised internal rate of return for dated cash flows (negative = paid in). None if undefined."""
    if not flows or not (any(v < 0 for _, v in flows) and any(v > 0 for _, v in flows)):
        return None
    d0 = flows[0][0]
    f = lambda r: sum(v / (1 + r) ** ((d - d0).days / 365.0) for d, v in flows)
    try:
        return bisect(f, -0.99, 1000.0, tol=1e-10)
    except ValueError:
        return None
