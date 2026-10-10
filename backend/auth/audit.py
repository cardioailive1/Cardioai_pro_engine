"""
CardioAI Pro — Persisted Audit Log
======================================
WHAT THIS CLOSES: the orchestrator's event log (orchestrator/orchestrator.py)
is an in-memory deque capped at 200 events, fanned out to the live
dashboard over WebSocket — real, but it is pipeline tracing for the
live view, not an audit record. It is lost on every restart and was
never meant to answer "who did what, when" after the fact. This module
is a real, persisted, queryable audit log, backed by the same Postgres
database as the RBAC tables (auth/models.py) — survives restarts,
survives redeploys, and can be queried months later.

WHAT IT RECORDS: two kinds of events, both real —
1. User actions: signup, login (success and failure), logout, approval/
   rejection/disabling of accounts — recorded from auth/routes.py with
   the acting user's identity, org, IP address and user agent.
2. Agent-pipeline activity: one row per completed or failed pipeline
   run (not one row per agent hop — that stays in the in-memory event
   log for the live dashboard; a row per task is what "audit controls:
   logging of agent activity" actually needs to answer "did this run,
   when, did it fail"), recorded from orchestrator/orchestrator.py.

RETENTION: no database-enforced TTL (Postgres has none built in without
an extension), so retention is enforced by `purge_older_than()` below,
run on a schedule — see the `cardioai-pro-audit-purge` cron service in
render.yaml, which calls scripts/purge_audit_log.py daily. Default
retention is 400 days (longer than a 1-year audit cycle, short of 2 —
adjust via AUDIT_LOG_RETENTION_DAYS if a specific SOC 2/HIPAA retention
requirement is set later).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Column, String, DateTime, Index, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from auth.models import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_RETENTION_DAYS = int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "400"))


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_org_created", "org_id", "created_at"),
        Index("ix_audit_log_action", "action"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False, index=True)

    # Who/what. Both nullable: a failed login attempt has no user_id yet
    # (the point is recording the attempt even though auth failed); a
    # system/pipeline event has no human actor at all.
    org_id = Column(UUID(as_uuid=False), nullable=True)
    user_id = Column(UUID(as_uuid=False), nullable=True)
    actor_label = Column(String(320), nullable=True)   # email or "system:<component>" — kept even if the user row is later deleted

    action = Column(String(100), nullable=False)        # e.g. "login.success", "login.failure", "user.approve", "pipeline.task_completed"
    resource_type = Column(String(100), nullable=True)  # e.g. "user", "pipeline_task", "document"
    resource_id = Column(String(200), nullable=True)

    status = Column(String(20), nullable=False, default="success")  # "success" | "failure"
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(500), nullable=True)
    event_metadata = Column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "created_at": self.created_at.isoformat() if self.created_at else None,
            "org_id": self.org_id, "user_id": self.user_id, "actor_label": self.actor_label,
            "action": self.action, "resource_type": self.resource_type, "resource_id": self.resource_id,
            "status": self.status, "ip_address": self.ip_address, "user_agent": self.user_agent,
            "metadata": self.event_metadata,
        }


def record_event(
    db: Session, *, action: str, status: str = "success",
    org_id: Optional[str] = None, user_id: Optional[str] = None, actor_label: Optional[str] = None,
    resource_type: Optional[str] = None, resource_id: Optional[str] = None,
    ip_address: Optional[str] = None, user_agent: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AuditLog:
    """
    Writes one audit row and commits it immediately — audit events are
    not rolled back alongside whatever business transaction they
    describe, since "the login attempt happened" is true whether or not
    the rest of the request later fails.
    """
    row = AuditLog(
        org_id=org_id, user_id=user_id, actor_label=actor_label, action=action,
        resource_type=resource_type, resource_id=resource_id, status=status,
        ip_address=ip_address, user_agent=user_agent, event_metadata=metadata,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def query_events(
    db: Session, *, org_id: Optional[str] = None, action: Optional[str] = None,
    limit: int = 100, before: Optional[datetime] = None,
) -> list[AuditLog]:
    q = db.query(AuditLog)
    if org_id is not None:
        q = q.filter(AuditLog.org_id == org_id)
    if action is not None:
        q = q.filter(AuditLog.action == action)
    if before is not None:
        q = q.filter(AuditLog.created_at < before)
    return q.order_by(AuditLog.created_at.desc()).limit(min(limit, 1000)).all()


def purge_older_than(db: Session, days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Deletes audit rows older than `days`. Returns the number of rows deleted. Called by scripts/purge_audit_log.py on the schedule configured in render.yaml."""
    cutoff = _now() - timedelta(days=days)
    deleted = db.query(AuditLog).filter(AuditLog.created_at < cutoff).delete(synchronize_session=False)
    db.commit()
    return deleted
