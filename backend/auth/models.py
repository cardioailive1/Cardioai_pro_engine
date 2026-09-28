"""
CardioAI Pro — RBAC Data Models
===================================
Organization-scoped users with roles, backed by real Postgres — not
in-memory, not local disk. Losing user accounts on a redeploy would be
a far more disruptive failure than losing demo documents ever was.

ROLE MODEL, DELIBERATELY SIMPLE RATHER THAN A FULL PERMISSION MATRIX:
- super_admin: the first person to sign up for a given organization
  name. Full access to all three gated surfaces (Main Engine, Clinician
  Dashboard, Payer Portal) plus user approval/management. Exactly one
  per org, set automatically on that first signup — never chosen by a
  user, never granted later.
- admin: promoted by a super_admin or another admin. Same surface
  access as super_admin, plus the same approval powers, but cannot be
  demoted/removed by anyone other than the super_admin.
- clinician: Main Engine + Clinician Dashboard only.
- payer_analyst: Payer Portal only.

WHAT THIS DOES NOT DO: this is real authentication and real
authorization, backed by a real database — a genuine upgrade from a
single shared HTTP Basic Auth credential. It is not a HIPAA compliance
program and not a SOC 2 control environment; those require legal
agreements, formal audits, and organizational policy work that no
amount of code produces on its own.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Enum as SAEnum, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    super_admin = "super_admin"
    admin = "admin"
    clinician = "clinician"
    payer_analyst = "payer_analyst"


class UserStatus(str, enum.Enum):
    pending = "pending"     # signed up, awaiting org admin approval — cannot log in yet
    active = "active"       # approved, can log in
    rejected = "rejected"   # explicitly denied by an admin — cannot log in, distinct from "never decided"
    disabled = "disabled"   # was active, later disabled by an admin — distinct from rejected (which implies never was active)


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(200), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("org_id", "email", name="uq_user_org_email"),
        Index("ix_users_email", "email"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    org_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    email = Column(String(320), nullable=False)
    password_hash = Column(String(200), nullable=False)
    full_name = Column(String(200), nullable=False)
    role = Column(SAEnum(Role, name="user_role"), nullable=False)
    status = Column(SAEnum(UserStatus, name="user_status"), nullable=False, default=UserStatus.pending)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    approved_by_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    accepted_legal_docs_at = Column(DateTime(timezone=True), nullable=True)  # single timestamp — all three (Terms, Privacy, BAA) are accepted together at signup, not tracked separately

    organization = relationship("Organization", back_populates="users")
    approved_by = relationship("User", remote_side=[id])

    def to_dict(self, include_email: bool = True) -> dict:
        d = {
            "id": self.id, "full_name": self.full_name, "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None, "created_at": self.created_at.isoformat() if self.created_at else None,
            "org_id": self.org_id,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "accepted_legal_docs_at": self.accepted_legal_docs_at.isoformat() if self.accepted_legal_docs_at else None,
        }
        if include_email:
            d["email"] = self.email
        return d


# Which of the three gated surfaces each role can reach. Data Room keeps
# its own separate password gate (unaffected by this — it predates RBAC
# and covers different, investor-facing content); every OTHER protected
# path just requires being an authenticated, active org member, and
# these three specifically also check the surface list below.
SURFACE_ACCESS: dict[str, set[Role]] = {
    "main_engine": {Role.super_admin, Role.admin, Role.clinician},
    "clinician_dashboard": {Role.super_admin, Role.admin, Role.clinician},
    "payer_portal": {Role.super_admin, Role.admin, Role.payer_analyst},
}

ADMIN_ROLES = {Role.super_admin, Role.admin}
