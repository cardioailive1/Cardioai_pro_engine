"""
CardioAI Pro — Enrollment Registry
========================================
Tracks actual covered membership from real X12 834 events
(integrations/x12_834.py) — who is currently enrolled, as opposed to
PopulationAggregator's "members with claims activity," which is a usage
proxy, not a coverage record. A member can be enrolled and generate zero
claims in a given period (the entire point of PMPM billing — you pay per
covered life, not per claim); a member can also show claims activity
from a period before their coverage terminated. These are genuinely
different populations, and billing/contracts.py's reconciliation can now
choose the real one (active_member_ids()) instead of only ever having
the usage proxy available.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from integrations.x12_834 import EnrollmentEvent, EnrollmentApplyResult, MAINTENANCE_ADD, MAINTENANCE_TERMINATE, MAINTENANCE_CHANGE, MAINTENANCE_REINSTATE


@dataclass
class EnrolledMember:
    member_id: str
    first_name: Optional[str]
    last_name: Optional[str]
    dob: Optional[str]
    sex: Optional[str]
    status: str  # "active" | "terminated"
    effective_date: Optional[str]
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id, "first_name": self.first_name, "last_name": self.last_name,
            "dob": self.dob, "sex": self.sex, "status": self.status,
            "effective_date": self.effective_date, "last_updated": self.last_updated,
        }


class EnrollmentRegistry:
    def __init__(self):
        self.members: dict[str, EnrolledMember] = {}

    def apply_events(self, events: list[EnrollmentEvent]) -> EnrollmentApplyResult:
        result = EnrollmentApplyResult()
        for e in events:
            if not e.member_id:
                result.skipped_no_member_id += 1
                continue

            if e.maintenance_type_code == MAINTENANCE_ADD:
                self.members[e.member_id] = EnrolledMember(
                    member_id=e.member_id, first_name=e.first_name, last_name=e.last_name,
                    dob=e.dob, sex=e.sex, status="active", effective_date=e.effective_date,
                )
                result.added.append(e.member_id)

            elif e.maintenance_type_code == MAINTENANCE_TERMINATE:
                existing = self.members.get(e.member_id)
                if existing:
                    existing.status = "terminated"
                    existing.effective_date = e.effective_date
                    existing.last_updated = time.time()
                else:
                    # A termination for a member we never saw an addition
                    # for — still record it as terminated rather than
                    # dropping the event, since real files can arrive
                    # out of full history (e.g. mid-contract onboarding).
                    self.members[e.member_id] = EnrolledMember(
                        member_id=e.member_id, first_name=e.first_name, last_name=e.last_name,
                        dob=e.dob, sex=e.sex, status="terminated", effective_date=e.effective_date,
                    )
                result.terminated.append(e.member_id)

            elif e.maintenance_type_code == MAINTENANCE_REINSTATE:
                existing = self.members.get(e.member_id)
                if existing:
                    existing.status = "active"
                    existing.effective_date = e.effective_date
                    existing.last_updated = time.time()
                else:
                    self.members[e.member_id] = EnrolledMember(
                        member_id=e.member_id, first_name=e.first_name, last_name=e.last_name,
                        dob=e.dob, sex=e.sex, status="active", effective_date=e.effective_date,
                    )
                result.reinstated.append(e.member_id)

            elif e.maintenance_type_code == MAINTENANCE_CHANGE:
                existing = self.members.get(e.member_id)
                if existing:
                    # A change updates details WITHOUT altering active/terminated status.
                    existing.first_name = e.first_name or existing.first_name
                    existing.last_name = e.last_name or existing.last_name
                    existing.dob = e.dob or existing.dob
                    existing.sex = e.sex or existing.sex
                    existing.last_updated = time.time()
                    result.changed.append(e.member_id)
                else:
                    # A change with no prior record — treat as an implicit
                    # add rather than silently discarding real member data.
                    self.members[e.member_id] = EnrolledMember(
                        member_id=e.member_id, first_name=e.first_name, last_name=e.last_name,
                        dob=e.dob, sex=e.sex, status="active", effective_date=e.effective_date,
                    )
                    result.added.append(e.member_id)

        return result

    def active_member_ids(self) -> set[str]:
        return {m.member_id for m in self.members.values() if m.status == "active"}

    def all_members(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in sorted(self.members.values(), key=lambda m: m.last_updated, reverse=True)]
