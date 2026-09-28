"""
CardioAI Pro — RBAC FastAPI Dependencies
=============================================
Extracts and validates the current user from the session cookie set at
login, for use as a FastAPI route dependency (`Depends(get_current_user)`,
`Depends(require_admin)`).

WHY A COOKIE, NOT A BEARER HEADER: a Bearer token only ever reaches the
server if a JavaScript fetch() call deliberately attaches it. A plain
browser navigation — typing the URL, following a bookmark, a page
reload — cannot attach a custom header at all. Gating page loads behind
a Bearer-only check would mean the pages could never actually load
except via JS-initiated requests, which isn't how a normal login
session works. An httpOnly cookie is sent automatically by the browser
on every request to the origin, including plain navigation, and is not
readable by JavaScript (httpOnly) or over plain HTTP if `secure` is set
— the standard, correct mechanism for this. main.py's ASGI middleware
reads the same cookie for page-level surface gating.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth.db import get_db
from auth.models import User, UserStatus, ADMIN_ROLES
from auth.security import decode_session_token

SESSION_COOKIE_NAME = "cardioai_session"
DATA_ROOM_COOKIE_NAME = "cardioai_dataroom"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(401, "Not logged in — no session cookie present.")
    payload = decode_session_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired session — please log in again.")
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user or user.status != UserStatus.active:
        raise HTTPException(401, "Session no longer valid — account may have been disabled since you logged in.")
    return user


def require_admin(current: User = Depends(get_current_user)) -> User:
    if current.role not in ADMIN_ROLES:
        raise HTTPException(403, "This action requires an admin or super_admin role.")
    return current
