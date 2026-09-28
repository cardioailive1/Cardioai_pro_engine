"""
CardioAI Pro — RBAC Database Connection
===========================================
Real Postgres via DATABASE_URL (Render auto-injects this from the
`databases:` block in render.yaml — nothing to configure by hand on a
real deploy). Falls back to a local SQLite file ONLY for environments
with no DATABASE_URL set at all (e.g. running this module's own tests
without a database available) — this fallback is not used in
production the way the R2/local-disk fallback is; render.yaml wires a
real database unconditionally, so a production deploy always has
DATABASE_URL set.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from auth.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./_auth_dev_fallback.db")

# Render's Postgres connection strings use the `postgres://` scheme;
# SQLAlchemy 2.x's psycopg2 dialect requires `postgresql://` — a very
# common, easy-to-miss gotcha when moving from Heroku/Render-style URLs.
# Also pin the driver explicitly (+psycopg2) — recent SQLAlchemy
# versions default a bare `postgresql://` to the psycopg (v3) dialect,
# which isn't installed here; psycopg2-binary is the one in
# requirements.txt, so the URL has to say so explicitly.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
