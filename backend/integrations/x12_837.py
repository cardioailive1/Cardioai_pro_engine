"""
CardioAI Pro — X12 837 Claims Parser
==========================================
Real X12 EDI parsing for professional (837P) and institutional (837I)
healthcare claims — the industry-standard format payers and clearinghouses
actually exchange, closing the biggest gap flagged earlier: the claims
pipeline previously only accepted a simplified `{member_age,
risk_flags_count}` JSON shape, nothing resembling a real claim.

HOW X12 DELIMITERS ACTUALLY WORK, AND WHY THAT MATTERS FOR CORRECTNESS:
X12 doesn't use fixed delimiters — the ISA segment (a FIXED-WIDTH 106
character header) declares them itself: the element separator is the
character immediately after "ISA", and the segment terminator is
whatever character follows the last field of the ISA line. A parser that
just assumes `~` and `*` (extremely common in the wild, and true of most
sample files) will silently misparse any real payer/clearinghouse file
that uses different characters. This parser reads the ISA line itself to
determine the real delimiters, rather than assuming.

SCOPE, STATED HONESTLY: X12 837's full structure uses nested HL
(Hierarchical Level) loops — Billing Provider -> Subscriber -> Patient —
with formal hierarchical parent/child linkage via HL01/HL02/HL03. This
parser does NOT implement full formal HL-loop hierarchy resolution.
Instead it does correct SEQUENTIAL CONTEXT TRACKING: it walks segments in
order, keeps "current subscriber" and "current claim" state, and updates
that state as it encounters NM1*IL (subscriber), CLM (claim), HI
(diagnosis), and SV1/SV2 (service line) segments — which correctly
handles the realistic, common case (segments appear in the order the
X12 837 implementation guide requires) without the added complexity of
resolving arbitrary HL parent/child nesting. A file with genuinely
irregular loop ordering could be misparsed by this simplification; a
production clearinghouse-grade parser would resolve HL loops formally.

Envelope/segment parsing (ISA delimiter detection, generic segment
splitting) lives in integrations/x12_common.py, shared with
integrations/x12_834.py (enrollment) — this module only handles the
837-specific segment interpretation (CLM/HI/SV1/SV2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from integrations.x12_common import X12Segment, parse_x12_envelope, parse_segments


@dataclass
class ParsedClaim:
    claim_id: Optional[str] = None
    subscriber_member_id: Optional[str] = None
    subscriber_last_name: Optional[str] = None
    subscriber_first_name: Optional[str] = None
    subscriber_dob: Optional[str] = None  # ISO date string
    subscriber_sex: Optional[str] = None
    diagnosis_codes: list[str] = field(default_factory=list)
    total_charge_amount: Optional[float] = None
    service_lines: list[dict[str, Any]] = field(default_factory=list)  # [{procedure_code, charge_amount}]

    def age_at(self, as_of: Optional[date] = None) -> Optional[int]:
        if not self.subscriber_dob:
            return None
        as_of = as_of or date.today()
        try:
            dob = datetime.strptime(self.subscriber_dob, "%Y%m%d").date()
        except ValueError:
            return None
        return as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))


# ICD-10 chapter ranges relevant to cardiovascular MACE risk — used to
# derive a real, clinically-grounded risk_flags_count from diagnosis
# codes, rather than an arbitrary number. I20-I25 = ischemic heart
# disease, I30-I52 = other heart disease (incl. heart failure, arrhythmia),
# I60-I69 = cerebrovascular disease (stroke) — the same event families
# training/label_schema.py's MACEEventType tracks.
CARDIOVASCULAR_ICD10_PREFIXES = tuple(
    [f"I{n}" for n in range(20, 26)] + [f"I{n}" for n in range(30, 53)] + [f"I{n}" for n in range(60, 70)]
)


def _is_cardiovascular_code(icd10: str) -> bool:
    return icd10.upper().startswith(CARDIOVASCULAR_ICD10_PREFIXES)


def parse_837_claims(raw: str) -> list[ParsedClaim]:
    """
    Walks the segment list with sequential context tracking (see module
    docstring for exactly what that means and doesn't cover) and returns
    one ParsedClaim per CLM segment encountered, carrying whichever
    subscriber context (NM1*IL) most recently preceded it.
    """
    segments = parse_segments(raw)
    claims: list[ParsedClaim] = []
    current_subscriber: dict[str, Any] = {}
    current_claim: Optional[ParsedClaim] = None

    for seg in segments:
        if seg.tag == "NM1" and seg.elements and seg.elements[0] == "IL":
            # NM1*IL*1*LASTNAME*FIRSTNAME****MI*MEMBERID
            current_subscriber = {
                "last_name": seg.elements[2] if len(seg.elements) > 2 else None,
                "first_name": seg.elements[3] if len(seg.elements) > 3 else None,
                "member_id": seg.elements[8] if len(seg.elements) > 8 else None,
            }

        elif seg.tag == "DMG" and current_claim is None:
            # DMG*D8*YYYYMMDD*SEX — demographic info following a subscriber NM1, before any CLM
            current_subscriber["dob"] = seg.elements[1] if len(seg.elements) > 1 else None
            current_subscriber["sex"] = seg.elements[2] if len(seg.elements) > 2 else None

        elif seg.tag == "CLM":
            if current_claim is not None:
                claims.append(current_claim)
            # CLM*CLAIMID*TOTALCHARGE*...
            claim_id = seg.elements[0] if len(seg.elements) > 0 else None
            total_charge = None
            if len(seg.elements) > 1:
                try:
                    total_charge = float(seg.elements[1])
                except ValueError:
                    total_charge = None
            current_claim = ParsedClaim(
                claim_id=claim_id, total_charge_amount=total_charge,
                subscriber_member_id=current_subscriber.get("member_id"),
                subscriber_last_name=current_subscriber.get("last_name"),
                subscriber_first_name=current_subscriber.get("first_name"),
                subscriber_dob=current_subscriber.get("dob"),
                subscriber_sex=current_subscriber.get("sex"),
            )

        elif seg.tag == "HI" and current_claim is not None:
            # HI*ABK:I2109*ABF:I509... — each element after the tag is qualifier:code
            for el in seg.elements:
                if ":" in el:
                    _, code = el.split(":", 1)
                    if code:
                        current_claim.diagnosis_codes.append(code)

        elif seg.tag in ("SV1", "SV2") and current_claim is not None:
            # SV1*HC:PROCEDURECODE*CHARGEAMOUNT*...
            proc = seg.elements[0].split(":", 1)[-1] if seg.elements else None
            charge = None
            if len(seg.elements) > 1:
                try:
                    charge = float(seg.elements[1])
                except ValueError:
                    charge = None
            current_claim.service_lines.append({"procedure_code": proc, "charge_amount": charge})

        elif seg.tag == "SE" and current_claim is not None:
            claims.append(current_claim)
            current_claim = None

    if current_claim is not None:
        claims.append(current_claim)

    return claims


def claim_to_cardioai_record(claim: ParsedClaim) -> dict[str, Any]:
    """
    Translates a ParsedClaim into the shape the existing claims pipeline
    already accepts (member_age, risk_flags_count) — plus real fields
    (member_id, total_charge_amount, diagnosis_codes) the placeholder
    shape never carried, which reports/cost_avoidance.py needs.
    """
    cv_dx_count = sum(1 for code in claim.diagnosis_codes if _is_cardiovascular_code(code))
    return {
        "member_id": claim.subscriber_member_id,
        "member_age": claim.age_at(),
        "risk_flags_count": min(cv_dx_count, 20),  # clamps to SCHEMAS["claims"]'s existing (0, 20) range
        "claim_id": claim.claim_id,
        "total_charge_amount": claim.total_charge_amount,
        "diagnosis_codes": claim.diagnosis_codes,
    }
