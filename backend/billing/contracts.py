"""
CardioAI Pro — PMPM Contract & Billing Administration
============================================================
Closes the gap flagged earlier: nothing in this engine tracked "members
covered under Payer X's contract" as a BILLING construct — only as a
risk-stratification population, which is a different concept. A payer's
population report can show 4,200 members with claims activity this month
without that number having any relationship to what they're actually
contracted (and being billed) for.

TWO DISTINCT MEMBER COUNTS, KEPT DELIBERATELY SEPARATE:
- CONTRACTED count: the negotiated number of covered lives in the PMPM
  agreement — a commercial/legal figure, set here explicitly, not
  derived from usage.
- RECONCILED count: how many distinct members actually had claims
  activity flow through the pipeline in a given billing period — an
  operational/usage figure, computed from real pipeline activity.
Real PMPM billing in the industry is USUALLY based on the contracted
count (you bill per covered life whether or not they generated a claim
that month, since PMPM is not fee-for-service) — but real contracts vary,
and a material gap between the two is exactly what a payer's own finance
team would want surfaced, not silently ignored. reconcile_membership()
below computes both and their gap explicitly.

WHAT THIS MODULE IS AND ISN'T: a real, working contract/invoice data
model and reconciliation/invoicing logic — not a full billing system
(no payment processing, no dunning, no real accounting-system
integration). It's the administrative record-keeping layer a payer
relationship needs that nothing in this engine had before.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

MEMBERSHIP_VARIANCE_ALERT_PCT = 10.0  # a reconciled count more than this % away from the contracted count gets flagged, not silently accepted


@dataclass
class Contract:
    contract_id: str
    payer_name: str
    pmpm_rate: float
    contracted_member_count: int
    start_date: date
    end_date: Optional[date] = None  # None = open-ended
    status: str = "active"  # "active" | "terminated"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "payer_name": self.payer_name, "pmpm_rate": self.pmpm_rate,
            "contracted_member_count": self.contracted_member_count,
            "start_date": self.start_date.isoformat(), "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status, "created_at": self.created_at,
        }


@dataclass
class Invoice:
    invoice_id: str
    contract_id: str
    period_label: str  # e.g. "2026-09"
    billing_basis: str  # "contracted" | "reconciled"
    billed_member_count: int
    pmpm_rate: float
    proration_factor: float  # 1.0 for a full period
    amount: float
    generated_at: float = field(default_factory=time.time)
    status: str = "pending"  # "pending" | "paid" | "void"

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id, "contract_id": self.contract_id, "period_label": self.period_label,
            "billing_basis": self.billing_basis, "billed_member_count": self.billed_member_count,
            "pmpm_rate": self.pmpm_rate, "proration_factor": round(self.proration_factor, 4),
            "amount": round(self.amount, 2), "generated_at": self.generated_at, "status": self.status,
        }


@dataclass
class ReconciliationResult:
    contract_id: str
    period_label: str
    contracted_count: int
    reconciled_count: int
    variance: int  # reconciled - contracted
    variance_pct: float
    alert: bool
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "period_label": self.period_label,
            "contracted_count": self.contracted_count, "reconciled_count": self.reconciled_count,
            "variance": self.variance, "variance_pct": round(self.variance_pct, 2),
            "alert": self.alert, "narrative": self.narrative,
        }


class ContractRegistry:
    def __init__(self):
        self.contracts: dict[str, Contract] = {}
        self.invoices: list[Invoice] = []

    def create_contract(
        self, payer_name: str, pmpm_rate: float, contracted_member_count: int,
        start_date: date, end_date: Optional[date] = None,
    ) -> Contract:
        contract = Contract(
            contract_id=f"CTR-{uuid.uuid4().hex[:8]}", payer_name=payer_name, pmpm_rate=pmpm_rate,
            contracted_member_count=contracted_member_count, start_date=start_date, end_date=end_date,
        )
        self.contracts[contract.contract_id] = contract
        return contract

    def terminate_contract(self, contract_id: str) -> Optional[Contract]:
        contract = self.contracts.get(contract_id)
        if contract:
            contract.status = "terminated"
        return contract

    def reconcile_membership(self, contract_id: str, period_label: str, actual_member_ids: set[str]) -> ReconciliationResult:
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"No contract {contract_id}")

        reconciled_count = len(actual_member_ids)
        contracted_count = contract.contracted_member_count
        variance = reconciled_count - contracted_count
        variance_pct = (variance / contracted_count * 100) if contracted_count else 0.0
        alert = abs(variance_pct) > MEMBERSHIP_VARIANCE_ALERT_PCT

        if alert:
            direction = "fewer" if variance < 0 else "more"
            narrative = (
                f"Reconciled member count ({reconciled_count}) is {abs(round(variance_pct, 1))}% {direction} than "
                f"the contracted count ({contracted_count}) for {period_label} — outside the "
                f"{MEMBERSHIP_VARIANCE_ALERT_PCT}% tolerance. Worth a real conversation with the payer before "
                f"invoicing on either number blindly."
            )
        else:
            narrative = f"Reconciled count ({reconciled_count}) is within {MEMBERSHIP_VARIANCE_ALERT_PCT}% of the contracted count ({contracted_count}) for {period_label}."

        return ReconciliationResult(
            contract_id=contract_id, period_label=period_label, contracted_count=contracted_count,
            reconciled_count=reconciled_count, variance=variance, variance_pct=variance_pct,
            alert=alert, narrative=narrative,
        )

    def generate_invoice(
        self, contract_id: str, period_label: str, billing_basis: str = "contracted",
        reconciled_count: Optional[int] = None, period_start: Optional[date] = None, period_end: Optional[date] = None,
    ) -> Invoice:
        """
        billing_basis="contracted" (the PMPM norm — bill per covered life
        regardless of utilization) or "reconciled" (bill on actual usage
        this period — some contracts are structured this way instead;
        requires reconciled_count). Prorates automatically if the
        contract's start/end date falls partway through the given period.
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"No contract {contract_id}")

        if billing_basis == "reconciled":
            if reconciled_count is None:
                raise ValueError("reconciled_count is required when billing_basis='reconciled'")
            billed_count = reconciled_count
        elif billing_basis == "contracted":
            billed_count = contract.contracted_member_count
        else:
            raise ValueError(f"billing_basis must be 'contracted' or 'reconciled', got {billing_basis!r}")

        proration = self._proration_factor(contract, period_start, period_end)
        amount = billed_count * contract.pmpm_rate * proration

        invoice = Invoice(
            invoice_id=f"INV-{uuid.uuid4().hex[:8]}", contract_id=contract_id, period_label=period_label,
            billing_basis=billing_basis, billed_member_count=billed_count, pmpm_rate=contract.pmpm_rate,
            proration_factor=proration, amount=amount,
        )
        self.invoices.append(invoice)
        return invoice

    def _proration_factor(self, contract: Contract, period_start: Optional[date], period_end: Optional[date]) -> float:
        """1.0 for a full period; a fraction if the contract's own start/end date truncates the covered period."""
        if period_start is None or period_end is None:
            return 1.0
        total_days = (period_end - period_start).days
        if total_days <= 0:
            return 1.0
        covered_start = max(period_start, contract.start_date)
        covered_end = min(period_end, contract.end_date) if contract.end_date else period_end
        covered_days = max(0, (covered_end - covered_start).days)
        return min(1.0, covered_days / total_days)

    def invoices_for_contract(self, contract_id: str) -> list[Invoice]:
        return [i for i in self.invoices if i.contract_id == contract_id]
