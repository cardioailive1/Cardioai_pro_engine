"""Cap table engine: holders, option pool, SAFE conversion and priced-round modelling.

Seeded with Cardio AI Corp's records (founder, 2026 Equity Incentive Plan grants, the two SAFEs). All share counts are
whole shares. Assumptions are flagged in `ASSUMPTIONS` so a reviewer can see what is not taken from a signed document.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .finmath import bisect

ASSUMPTIONS = [
    "Source: CAP_TABLE_V6A (in the Document Room). The pre-seed round is a proposal still under negotiation; every after-round figure is a projection at $2.00 per share.",
    "Grant dates for the four option grants are set to 5 Oct 2026 (Board consent effective date); confirm against the signed grant agreements.",
    "SAFEs convert in the next priced round at the lower of the discount price and the post-money cap price; "
    "the post-money capitalization includes the whole option pool and the converting SAFEs.",
    "The Mayo Clinic Platform and Dr. Nahar SAFEs have not converted; they stay unconverted in today's table. Pool shares not yet granted are counted in the fully diluted share count (as in the 409A report and the cap table workbook).",
]


@dataclass
class Safe:
    holder: str
    purchase: float
    discount: float = 0.20
    cap: Optional[float] = 50_000_000.0
    post_money: bool = True
    date: Optional[date] = None


@dataclass
class Grant:
    holder: str
    shares: int
    strike: float
    grant_date: date
    kind: str = "NSO"           # "ISO" (any excess over the $100K yearly limit becomes NSO by law) or "NSO"
    role: str = ""
    cliff_months: int = 12
    vest_months: int = 48
    fully_vested_at_grant: bool = False
    term_years: float = 10.0
    expected_term_years: float = 6.25


@dataclass
class CapTable:
    company: str = "Cardio AI Corp"
    authorized_common: int = 25_000_000
    par: float = 1.00
    common: dict = field(default_factory=dict)     # holder -> issued shares
    grants: list = field(default_factory=list)
    safes: list = field(default_factory=list)
    pool_reserve: int = 0
    preferred: dict = field(default_factory=dict)  # holder -> shares (none today)

    # ---- counts -------------------------------------------------------------------------------------------------
    @property
    def common_total(self) -> int:
        return int(sum(self.common.values()) + sum(self.preferred.values()))

    @property
    def granted_total(self) -> int:
        return int(sum(g.shares for g in self.grants))

    @property
    def pool_available(self) -> int:
        return int(self.pool_reserve - self.granted_total)

    # ---- SAFE conversion ----------------------------------------------------------------------------------------
    def convert_safes(self, round_price: float) -> list[dict]:
        """Convert every SAFE at the lower of discount price and cap price. Returns one dict per SAFE."""
        base = self.common_total + self.pool_reserve
        shares = [0.0] * len(self.safes)
        for _ in range(200):
            new = []
            for s in self.safes:
                cap_shares = base + sum(shares) if s.post_money else base
                cap_price = s.cap / cap_shares if s.cap else math.inf
                disc_price = round_price * (1 - s.discount)
                new.append(s.purchase / min(cap_price, disc_price))
            done = all(abs(a - b) < 1e-9 for a, b in zip(new, shares))
            shares = new
            if done:
                break
        out = []
        for s, n in zip(self.safes, shares):
            cap_shares = base + sum(shares) if s.post_money else base
            cap_price = s.cap / cap_shares if s.cap else math.inf
            disc_price = round_price * (1 - s.discount)
            use_cap = cap_price < disc_price
            price = cap_price if use_cap else disc_price
            out.append({"holder": s.holder, "purchase": s.purchase, "price": round(price, 6),
                        "basis": "cap" if use_cap else "discount", "shares": int(round(n)),
                        "cap_price": round(cap_price, 6) if math.isfinite(cap_price) else None,
                        "discount_price": round(disc_price, 6)})
        return out

    def safe_cap_price(self) -> Optional[float]:
        """Price per share the cap alone would give (independent of the round price)."""
        if not self.safes:
            return None
        return self.convert_safes(round_price=1e9)[0]["price"]

    # ---- round modelling ----------------------------------------------------------------------------------------
    def model_round(self, *, investment: float, price: float | None = None, pre_money: float | None = None,
                    pool_increase: int = 0, investor: str = "New investors") -> dict:
        """A priced round. Give the price per share OR the pre-money valuation (pre-money counts the SAFE shares and any
        pool increase, as is usual). Returns the post-round table and ownership."""
        if (price is None) == (pre_money is None):
            raise ValueError("give exactly one of price or pre_money")

        def pre_fd(p: float) -> float:
            return self.common_total + self.pool_reserve + pool_increase + sum(c["shares"] for c in self.convert_safes(p))

        if price is None:
            price = bisect(lambda p: p * pre_fd(p) - pre_money, 1e-6, 1e6)
        conv = self.convert_safes(price)
        pre_shares = int(self.common_total + self.pool_reserve + pool_increase + sum(c["shares"] for c in conv))
        new_shares = int(round(investment / price))
        post_shares = pre_shares + new_shares
        rows = []
        for h, n in self.common.items():
            rows.append({"holder": h, "class": "Common", "shares": int(n)})
        for h, n in self.preferred.items():
            rows.append({"holder": h, "class": "Preferred", "shares": int(n)})
        for g in self.grants:
            rows.append({"holder": g.holder, "class": "Options (granted)", "shares": g.shares})
        rows.append({"holder": "Option pool (available)", "class": "Pool", "shares": self.pool_available + pool_increase})
        for c in conv:
            rows.append({"holder": c["holder"], "class": "SAFE (converted)", "shares": c["shares"]})
        rows.append({"holder": investor, "class": "Preferred (new)", "shares": new_shares})
        for r in rows:
            r["pct"] = r["shares"] / post_shares
        return {"price": round(price, 6), "investment": investment, "pre_money": round(price * pre_shares, 2),
                "post_money": round(price * post_shares, 2), "pre_money_shares": pre_shares, "post_money_shares": post_shares,
                "new_shares": new_shares, "pool_increase": pool_increase, "investor": investor, "safes": conv, "rows": rows,
                "authorized_remaining": self.authorized_common - post_shares,
                "authorized_remaining_pre_round": self.authorized_common - pre_shares}

    def after_round(self, r: dict) -> "CapTable":
        """The cap table once a modelled round (the dict from model_round) has closed: SAFEs converted to preferred, new
        investors added, pool increased. Used to stack a later round on an earlier one."""
        new = CapTable(company=self.company, authorized_common=self.authorized_common, par=self.par, common=dict(self.common),
                       grants=list(self.grants), safes=[], pool_reserve=self.pool_reserve + r["pool_increase"], preferred=dict(self.preferred))
        for c in r["safes"]:
            new.preferred[f"{c['holder']} (converted SAFE)"] = c["shares"]
        new.preferred[r["investor"]] = new.preferred.get(r["investor"], 0) + r["new_shares"]
        return new

    def current_ownership(self) -> list[dict]:
        """Today's fully diluted table, SAFEs not yet converted."""
        total = self.common_total + self.pool_reserve
        rows = [{"holder": h, "class": "Common", "shares": int(n)} for h, n in self.common.items()]
        rows += [{"holder": g.holder, "class": "Options (granted)", "shares": g.shares} for g in self.grants]
        rows.append({"holder": "Option pool (available)", "class": "Pool", "shares": self.pool_available})
        for r in rows:
            r["pct"] = r["shares"] / total
        return rows

    def to_dict(self) -> dict:
        return {"company": self.company, "authorized_common": self.authorized_common, "par": self.par,
                "issued": self.common_total, "pool_reserve": self.pool_reserve, "granted": self.granted_total,
                "pool_available": self.pool_available,
                "holders": [{"holder": h, "shares": n} for h, n in self.common.items()],
                "grants": [{**g.__dict__, "grant_date": g.grant_date.isoformat()} for g in self.grants],
                "safes": [{**s.__dict__, "date": s.date.isoformat() if s.date else None} for s in self.safes],
                "assumptions": ASSUMPTIONS}


def seed_cap_table() -> CapTable:
    gd = date(2026, 10, 5)
    t = CapTable(common={"Sampson Kontomah (Founder)": 12_750_000}, pool_reserve=3_108_696)
    t.grants = [
        Grant("Everlyn Ndirangu", 594_348, 1.00, gd, "ISO", "CFO"),
        Grant("Galax Womack", 594_348, 1.00, gd, "ISO", "CTO"),
        Grant("Avi Patel", 445_761, 1.00, gd, "ISO", "VP Engineering"),
        Grant("Dr. Tamanna Nahar", 400_000, 1.00, gd, "NSO", "Chief Medical Advisor",
              fully_vested_at_grant=True, expected_term_years=10.0),
    ]
    t.safes = [Safe("Mayo Clinic Platform", 800_000, date=date(2026, 10, 5)), Safe("Dr. Tamanna Nahar", 30_000, date=date(2024, 1, 1))]
    return t
