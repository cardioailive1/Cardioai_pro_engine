"""
CardioAI Pro — Care Management Tasks
==========================================
Closes the gap named directly: after a care manager sees a member's risk
score in the payer portal's drawer, nothing tracked what happened next —
no outreach record, no enrollment, no referral, nothing. A human had to
remember to act, and the system had no way to know whether they did.

WHY THIS ALSO IMPROVES reports/cost_avoidance.py, NOT JUST THE UI:
that module's pre/post cost-trend comparison anchors on `flagged_at` —
when the risk score crossed the threshold — as a proxy for "when
intervention started." That's a real, named weakness in that module's own
docstring: a score crossing a threshold isn't the same event as someone
actually doing something about it. A CareTask's `intervention_started_at`
(set when a task moves past outreach into an actual enrollment/referral)
is a genuinely better anchor — it's the closest thing this system has to
"an intervention actually happened here, at this real moment." See
get_intervention_anchor() below, and its use in api/routes.py's
cost-avoidance endpoint.

STATUS LIFECYCLE, DELIBERATELY LINEAR: pending -> contacted ->
(enrolled | declined | referred). No backward transitions modeled — a
declined member who later re-engages gets a new task, not a status
reopened on the old one, so the history stays an honest record of what
actually happened at each point, not a mutated single row.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

VALID_TASK_TYPES = {"outreach_call", "enrollment", "referral", "care_coordination"}
VALID_STATUSES = {"pending", "contacted", "enrolled", "declined", "referred"}
TERMINAL_STATUSES = {"enrolled", "declined", "referred"}  # a task in one of these has resolved — its updated_at is the closest real "intervention happened" timestamp this system has


@dataclass
class CareTask:
    task_id: str
    member_id: str
    task_type: str
    status: str
    assigned_to: str
    notes: str
    created_at: float
    updated_at: float
    status_history: list[dict[str, Any]] = field(default_factory=list)  # [{status, changed_at}, ...] — the real audit trail

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "member_id": self.member_id, "task_type": self.task_type,
            "status": self.status, "assigned_to": self.assigned_to, "notes": self.notes,
            "created_at": self.created_at, "updated_at": self.updated_at, "status_history": self.status_history,
        }


class CareTaskRegistry:
    def __init__(self):
        self.tasks: dict[str, CareTask] = {}

    def create_task(self, member_id: str, task_type: str, assigned_to: str, notes: str = "") -> tuple[Optional[CareTask], Optional[str]]:
        if task_type not in VALID_TASK_TYPES:
            return None, f"task_type must be one of {sorted(VALID_TASK_TYPES)}, got {task_type!r}"
        now = time.time()
        task = CareTask(
            task_id=f"TASK-{uuid.uuid4().hex[:8]}", member_id=member_id, task_type=task_type,
            status="pending", assigned_to=assigned_to, notes=notes, created_at=now, updated_at=now,
            status_history=[{"status": "pending", "changed_at": now}],
        )
        self.tasks[task.task_id] = task
        return task, None

    def update_status(self, task_id: str, new_status: str) -> tuple[Optional[CareTask], Optional[str]]:
        if new_status not in VALID_STATUSES:
            return None, f"status must be one of {sorted(VALID_STATUSES)}, got {new_status!r}"
        task = self.tasks.get(task_id)
        if not task:
            return None, f"No task {task_id}"
        if task.status in TERMINAL_STATUSES:
            return None, f"Task {task_id} is already {task.status} (terminal) — create a new task for renewed outreach instead of reopening this one, so the history stays an honest record."
        now = time.time()
        task.status = new_status
        task.updated_at = now
        task.status_history.append({"status": new_status, "changed_at": now})
        return task, None

    def tasks_for_member(self, member_id: str) -> list[CareTask]:
        return [t for t in self.tasks.values() if t.member_id == member_id]

    def all_tasks(self, status_filter: Optional[str] = None) -> list[CareTask]:
        tasks = list(self.tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return sorted(tasks, key=lambda t: t.updated_at, reverse=True)

    def get_intervention_anchor(self, member_id: str) -> Optional[float]:
        """
        Returns the timestamp a member's cost trend should actually
        anchor on: the EARLIEST terminal-status task for this member
        (enrolled/declined/referred — the point something real actually
        happened), or None if no task has reached a terminal state yet.
        Callers (see api/routes.py) fall back to flagged_at when this
        returns None, rather than blocking the report on it.
        """
        terminal_tasks = [t for t in self.tasks_for_member(member_id) if t.status in TERMINAL_STATUSES]
        if not terminal_tasks:
            return None
        return min(t.updated_at for t in terminal_tasks)
