"""
CardioAI Pro — X12 834 Enrollment Parser
==============================================
Real X12 834 (Benefit Enrollment and Maintenance) parsing — the actual
EDI standard payers use to communicate who is enrolled, added, changed,
or terminated from a plan. Closes a real, named gap: X12 837 (claims)
tells you who USED services; it says nothing about who is actually
COVERED. `contracted_member_count` reconciliation
(billing/contracts.py) was, until this existed, only ever checked
against members with claims ACTIVITY — a usage proxy, not a real
enrollment count. PMPM billing is specifically NOT usage-based (you pay
per covered life whether or not they submitted a claim that month), so
reconciling against a usage proxy was always the wrong comparison when
real enrollment data is available — this is what makes that comparison
real instead of approximate.

KEY SEGMENT: INS — the member-level detail segment. INS03 (this
parser's `maintenance_type_code`) is what actually matters:
  021 = Addition (member is newly enrolled)
  024 = Termination (member's coverage ends)
  001 = Change (member's enrollment details changed — still covered)
  025 = Reinstatement (member re-added after a termination)
Real X12 834 implementation guides define more codes than this; these
four cover the realistic majority of files and are what
apply_enrollment_events() below actually acts on. An unrecognized code
is surfaced, not silently dropped — see the "unrecognized" note in
parse_834_enrollment()'s return.

SAME SCOPE CAVEAT AS x12_837.py: sequential context tracking (walking
segments in the order a real 834 implementation guide requires), not
full formal HL-loop hierarchy resolution. See that module's docstring
for what that distinction means and doesn't cover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from integrations.x12_common import parse_segments

MAINTENANCE_ADD = "021"
MAINTENANCE_TERMINATE = "024"
MAINTENANCE_CHANGE = "001"
MAINTENANCE_REINSTATE = "025"
KNOWN_MAINTENANCE_CODES = {MAINTENANCE_ADD, MAINTENANCE_TERMINATE, MAINTENANCE_CHANGE, MAINTENANCE_REINSTATE}


@dataclass
class EnrollmentEvent:
    member_id: Optional[str] = None
    maintenance_type_code: Optional[str] = None
    last_name: Optional[str] = None
    first_name: Optional[str] = None
    dob: Optional[str] = None  # ISO-ish YYYYMMDD, as X12 carries it
    sex: Optional[str] = None
    effective_date: Optional[str] = None  # YYYYMMDD — from DTP*348 (coverage begin) or DTP*349 (coverage end), whichever applies


def parse_834_enrollment(raw: str) -> tuple[list[EnrollmentEvent], list[str]]:
    """
    Returns (events, unrecognized_maintenance_codes). Walks segments with
    the same sequential-context-tracking approach x12_837.py uses: each
    INS segment starts a new member-event context; subsequent NM1/DMG/REF/
    DTP segments before the next INS populate that event.
    """
    segments = parse_segments(raw)
    events: list[EnrollmentEvent] = []
    unrecognized: list[str] = []
    current: Optional[EnrollmentEvent] = None

    for seg in segments:
        if seg.tag == "INS":
            if current is not None:
                events.append(current)
            # INS*Y*18*021*A~ — INS03 (index 2 in elements, since elements excludes the tag) is the maintenance type code
            code = seg.elements[2] if len(seg.elements) > 2 else None
            if code and code not in KNOWN_MAINTENANCE_CODES:
                unrecognized.append(code)
            current = EnrollmentEvent(maintenance_type_code=code)

        elif seg.tag == "REF" and current is not None and seg.elements and seg.elements[0] in ("0F", "1L"):
            # REF*0F*MEMBERID~ (0F = subscriber number) or REF*1L*MEMBERID~ (group/policy number context varies by trading partner)
            current.member_id = seg.elements[1] if len(seg.elements) > 1 else current.member_id

        elif seg.tag == "NM1" and current is not None and seg.elements and seg.elements[0] == "IL":
            current.last_name = seg.elements[2] if len(seg.elements) > 2 else None
            current.first_name = seg.elements[3] if len(seg.elements) > 3 else None
            if len(seg.elements) > 8 and not current.member_id:
                current.member_id = seg.elements[8]

        elif seg.tag == "DMG" and current is not None:
            current.dob = seg.elements[1] if len(seg.elements) > 1 else None
            current.sex = seg.elements[2] if len(seg.elements) > 2 else None

        elif seg.tag == "DTP" and current is not None and seg.elements:
            # DTP*348*D8*20260101~ (348 = coverage begin) or DTP*349 (coverage end)
            if seg.elements[0] in ("348", "349") and len(seg.elements) > 2:
                current.effective_date = seg.elements[2]

    if current is not None:
        events.append(current)

    return events, unrecognized


@dataclass
class EnrollmentApplyResult:
    added: list[str] = field(default_factory=list)
    terminated: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    reinstated: list[str] = field(default_factory=list)
    skipped_no_member_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added, "terminated": self.terminated, "changed": self.changed,
            "reinstated": self.reinstated, "skipped_no_member_id": self.skipped_no_member_id,
        }
