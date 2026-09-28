"""
CardioAI Pro — RBAC API Routes
===================================
Signup (first user per org becomes super_admin automatically; every
subsequent signup for that same org is created as pending and cannot
log in until an admin approves it), login, and the approval workflow
admins use to review pending signups.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from auth.db import get_db
from auth.models import Organization, User, Role, UserStatus, ADMIN_ROLES
from auth.security import hash_password, verify_password, create_session_token, JWT_EXPIRE_HOURS
from auth.deps import get_current_user, require_admin, SESSION_COOKIE_NAME

router = APIRouter(prefix="/auth", tags=["auth"])

REQUESTABLE_ROLES = {Role.admin, Role.clinician, Role.payer_analyst}  # super_admin is never requested — only assigned automatically to an org's first user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, httponly=True, samesite="lax",
        secure=True,  # HTTPS only — Render serves everything over HTTPS, including local health checks proxied through it
        max_age=JWT_EXPIRE_HOURS * 3600, path="/",
    )


class SignupRequest(BaseModel):
    org_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    full_name: str = Field(min_length=1, max_length=200)
    requested_role: Role
    accepted_legal_docs: bool = False


class LoginRequest(BaseModel):
    org_name: str
    email: EmailStr
    password: str


class ApproveRequest(BaseModel):
    role: Role | None = None  # admin may adjust the role at approval time; None keeps what was requested


@router.post("/signup")
def signup(body: SignupRequest, response: Response, db: Session = Depends(get_db)):
    if not body.accepted_legal_docs:
        raise HTTPException(400, "You must accept the Terms of Use, Privacy Statement, and Business Associate Agreement to create an account.")
    if body.requested_role not in REQUESTABLE_ROLES:
        raise HTTPException(400, f"requested_role must be one of {[r.value for r in REQUESTABLE_ROLES]} — super_admin is assigned automatically, never requested")

    from auth.models import _now
    org = db.query(Organization).filter(func.lower(Organization.name) == body.org_name.strip().lower()).first()

    if org is None:
        # First person for this org name — becomes super_admin, active immediately, no approval needed.
        org = Organization(name=body.org_name.strip())
        db.add(org)
        db.flush()  # get org.id without committing yet
        user = User(
            org_id=org.id, email=str(body.email).lower(), password_hash=hash_password(body.password),
            full_name=body.full_name, role=Role.super_admin, status=UserStatus.active,
            accepted_legal_docs_at=_now(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        token = create_session_token(user.id, org.id, user.role.value)
        _set_session_cookie(response, token)
        return {
            "created_org": True, "status": "active", "role": user.role.value,
            "message": f'Created new organization "{org.name}" and made you its super admin — you can log in immediately.',
            "user": user.to_dict(),
        }

    existing = db.query(User).filter(User.org_id == org.id, User.email == str(body.email).lower()).first()
    if existing:
        raise HTTPException(409, "An account with this email already exists in this organization.")

    user = User(
        org_id=org.id, email=str(body.email).lower(), password_hash=hash_password(body.password),
        full_name=body.full_name, role=body.requested_role, status=UserStatus.pending,
        accepted_legal_docs_at=_now(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "created_org": False, "status": "pending", "role": user.role.value,
        "message": f'Account created for "{org.name}" as {user.role.value}, routed to your organization\'s admin for approval. You cannot log in until approved.',
        "user": user.to_dict(),
    }


@router.post("/login")
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    org = db.query(Organization).filter(func.lower(Organization.name) == body.org_name.strip().lower()).first()
    user = None
    if org:
        user = db.query(User).filter(User.org_id == org.id, User.email == str(body.email).lower()).first()

    # Deliberately generic on wrong org/email/password — do not reveal which part was wrong (avoids user enumeration).
    if not org or not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid organization, email, or password.")

    if user.status != UserStatus.active:
        reason = {
            UserStatus.pending: "Your account is still awaiting approval from your organization's admin.",
            UserStatus.rejected: "Your account request was not approved.",
            UserStatus.disabled: "Your account has been disabled. Contact your organization's admin.",
        }[user.status]
        raise HTTPException(403, reason)

    token = create_session_token(user.id, org.id, user.role.value)
    _set_session_cookie(response, token)
    return {"user": user.to_dict()}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"logged_out": True}


@router.get("/me")
def me(current: User = Depends(get_current_user)):
    return current.to_dict()


@router.get("/pending")
def list_pending(current: User = Depends(require_admin), db: Session = Depends(get_db)):
    pending = db.query(User).filter(User.org_id == current.org_id, User.status == UserStatus.pending).order_by(User.created_at).all()
    return {"pending": [u.to_dict() for u in pending]}


@router.get("/org/users")
def list_org_users(current: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).filter(User.org_id == current.org_id).order_by(User.created_at).all()
    return {"users": [u.to_dict() for u in users]}


@router.post("/approve/{user_id}")
def approve_user(user_id: str, body: ApproveRequest, current: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id, User.org_id == current.org_id).first()
    if not target:
        raise HTTPException(404, "No such user in your organization.")
    if target.status != UserStatus.pending:
        raise HTTPException(400, f"User is not pending (current status: {target.status.value}).")
    if body.role is not None:
        if body.role not in REQUESTABLE_ROLES:
            raise HTTPException(400, f"role must be one of {[r.value for r in REQUESTABLE_ROLES]}")
        target.role = body.role
    from auth.models import _now
    target.status = UserStatus.active
    target.approved_by_id = current.id
    target.approved_at = _now()
    db.commit()
    db.refresh(target)
    return {"approved": True, "user": target.to_dict()}


@router.post("/reject/{user_id}")
def reject_user(user_id: str, current: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id, User.org_id == current.org_id).first()
    if not target:
        raise HTTPException(404, "No such user in your organization.")
    if target.status != UserStatus.pending:
        raise HTTPException(400, f"User is not pending (current status: {target.status.value}).")
    target.status = UserStatus.rejected
    db.commit()
    return {"rejected": True, "user_id": user_id}


@router.post("/users/{user_id}/disable")
def disable_user(user_id: str, current: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id, User.org_id == current.org_id).first()
    if not target:
        raise HTTPException(404, "No such user in your organization.")
    if target.role == Role.super_admin:
        raise HTTPException(400, "The organization's super_admin cannot be disabled.")
    if target.id == current.id:
        raise HTTPException(400, "You cannot disable your own account.")
    return {"disabled": True, "user_id": user_id}
