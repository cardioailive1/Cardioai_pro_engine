"""
CardioAI Pro — RBAC Security Primitives
============================================
Password hashing (bcrypt via passlib) and session tokens (JWT via
python-jose). Two real, standard, well-understood mechanisms — not
homegrown crypto.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 12


def _secret_key() -> str:
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. render.yaml generates this automatically on deploy "
            "(generateValue: true) — if you're seeing this locally, set it yourself for testing."
        )
    return key


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_session_token(user_id: str, org_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": user_id, "org_id": org_id, "role": role, "exp": expire}
    return jwt.encode(payload, _secret_key(), algorithm=JWT_ALGORITHM)


DATA_ROOM_TOKEN_HOURS = 12


def create_data_room_token() -> str:
    """
    A separate, single-shared-password access token for the Data Room —
    deliberately NOT tied to any org/user identity, unlike
    create_session_token above. Due-diligence investors come from many
    different firms, not one organization's account system; they share
    one access code the same way a real investor data room typically
    works, and this token just proves "this browser supplied the
    correct code," nothing more. Reuses the same signing key and
    verification path as the RBAC session token (decode_session_token
    works on either), distinguished by the "scope" claim instead of a
    "sub" — the Data Room's own gate checks for that claim specifically,
    so an RBAC session token can't be reused here and vice versa.
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=DATA_ROOM_TOKEN_HOURS)
    payload = {"scope": "data_room_access", "exp": expire}
    return jwt.encode(payload, _secret_key(), algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _secret_key(), algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None
