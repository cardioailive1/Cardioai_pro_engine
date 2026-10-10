"""
CardioAI Pro — Audit Log Retention Purge
=============================================
Deletes audit_log rows older than AUDIT_LOG_RETENTION_DAYS (default 400
— see auth/audit.py). Run on a schedule by the `cardioai-pro-audit-purge`
cron service defined in render.yaml; also safe to run manually:

    python -m scripts.purge_audit_log
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth.db import SessionLocal, init_db
from auth.audit import purge_older_than, DEFAULT_RETENTION_DAYS


def main() -> None:
    init_db()  # idempotent; ensures audit_log exists even if this runs before the web service's first startup
    db = SessionLocal()
    try:
        deleted = purge_older_than(db, days=DEFAULT_RETENTION_DAYS)
        print(f"Purged {deleted} audit_log row(s) older than {DEFAULT_RETENTION_DAYS} days.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
