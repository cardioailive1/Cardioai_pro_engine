"""
Keeps the engine's working data across restarts.

Before this, patients, care stages, payer members, care tasks, contracts and enrollment lived only in memory and cleared on every
restart or deploy. This saves them to the same database as the accounts, and loads them back at startup.

How: a background task takes a snapshot every ENGINE_STATE_SAVE_SECONDS (default 10 s) and writes only the records that changed
(SHA-256 of each record's JSON), and deletes records that no longer exist. A final save runs on shutdown. A crash can lose up to one
interval of changes. Each record is encrypted with the application key (encryption.py) before it is stored, so the database holds
no readable patient data.

On by default when a real database is configured (DATABASE_URL, not the local SQLite fallback); ENGINE_STATE_PERSISTENCE=1 or 0 overrides.

LIMITS: ONE web instance only. Two instances would each overwrite the other's state. Latest 10-second ECG per patient is kept (about
0.3 MB each). The model event log and the quality / trend caches are not saved (they rebuild from new data).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import fields, asdict
from datetime import date, datetime, timezone
from typing import Any, Callable

import numpy as np
from sqlalchemy import Column, DateTime, String, Text

from auth.models import Base
from encryption import decrypt_str, encrypt_str

log = logging.getLogger("engine_state")


class EngineRecord(Base):
    __tablename__ = "engine_records"
    store = Column(String(40), primary_key=True)
    record_id = Column(String(200), primary_key=True)
    payload = Column(Text, nullable=False)                  # encrypted JSON
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"cannot store {type(o).__name__}")


def _plain(obj) -> dict:
    """Shallow, array-free field dict of a dataclass (asdict would deep-copy the ECG array)."""
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


# ---- per-store adapters: (records dict, dump -> dict, load dict -> object) ----
def build_adapters(o) -> dict[str, dict[str, Any]]:
    from orchestrator.patient_registry import PatientRecord
    from orchestrator.care_continuum import PatientContinuum, CareStage
    from reports.population_report import MemberRiskRecord
    from care_management.tasks import CareTask
    from billing.enrollment import EnrolledMember
    from billing.contracts import Contract, Invoice

    def dump_patient(r):
        d = _plain(r); d.pop("latest_ecg"); return d

    def load_patient(d):
        return PatientRecord(**d)

    def dump_continuum(c):
        return {"patient_id": c.patient_id, "stage": c.stage.value, "history": c.history, "last_updated": c.last_updated}

    def load_continuum(d):
        return PatientContinuum(patient_id=d["patient_id"], stage=CareStage(d["stage"]), history=d["history"], last_updated=d["last_updated"])

    def dump_contract(c):
        d = asdict(c); d["start_date"] = c.start_date.isoformat(); d["end_date"] = c.end_date.isoformat() if c.end_date else None; return d

    def load_contract(d):
        d = dict(d); d["start_date"] = date.fromisoformat(d["start_date"]); d["end_date"] = date.fromisoformat(d["end_date"]) if d.get("end_date") else None
        return Contract(**d)

    return {
        "patients": {"items": lambda: o.patient_registry.patients, "dump": dump_patient, "load": load_patient,
                     "put": lambda k, v: o.patient_registry.patients.__setitem__(k, v)},
        "continuum": {"items": lambda: o.continuum_tracker.patients, "dump": dump_continuum, "load": load_continuum,
                      "put": lambda k, v: o.continuum_tracker.patients.__setitem__(k, v)},
        "payer_members": {"items": lambda: o.population_aggregator.members, "dump": asdict, "load": lambda d: MemberRiskRecord(**d),
                          "put": lambda k, v: o.population_aggregator.members.__setitem__(k, v)},
        "care_tasks": {"items": lambda: o.care_task_registry.tasks, "dump": asdict, "load": lambda d: CareTask(**d),
                       "put": lambda k, v: o.care_task_registry.tasks.__setitem__(k, v)},
        "enrollment": {"items": lambda: o.enrollment_registry.members, "dump": asdict, "load": lambda d: EnrolledMember(**d),
                       "put": lambda k, v: o.enrollment_registry.members.__setitem__(k, v)},
        "contracts": {"items": lambda: o.contract_registry.contracts, "dump": dump_contract, "load": load_contract,
                      "put": lambda k, v: o.contract_registry.contracts.__setitem__(k, v)},
        "invoices": {"items": lambda: {i.invoice_id: i for i in o.contract_registry.invoices}, "dump": asdict, "load": lambda d: Invoice(**d),
                     "put": lambda k, v: o.contract_registry.invoices.append(v)},
    }


def persistence_enabled() -> bool:
    flag = os.environ.get("ENGINE_STATE_PERSISTENCE", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    from auth.db import DATABASE_URL
    return bool(os.environ.get("DATABASE_URL")) and not DATABASE_URL.startswith("sqlite")


class StatePersister:
    def __init__(self, orchestrator, session_factory: Callable, interval_s: float | None = None):
        self.o = orchestrator
        self.session_factory = session_factory
        self.interval_s = interval_s if interval_s is not None else float(os.environ.get("ENGINE_STATE_SAVE_SECONDS", "10"))
        self.adapters = build_adapters(orchestrator)
        self._hashes: dict[tuple[str, str], str] = {}
        self._ecg_stamp: dict[str, float] = {}
        self.last_saved_at: float | None = None
        self.last_error: str | None = None
        self.loaded: dict[str, int] = {}

    # ---------------- load ----------------
    def load(self) -> dict[str, int]:
        counts, skipped = {}, 0
        with self.session_factory() as db:
            rows = db.query(EngineRecord).all()
        by_store: dict[str, list] = {}
        for r in rows:
            by_store.setdefault(r.store, []).append(r)
        for store, ad in self.adapters.items():
            n = 0
            for r in sorted(by_store.get(store, []), key=lambda x: x.updated_at or datetime.min.replace(tzinfo=timezone.utc)):
                try:
                    raw = decrypt_str(r.payload)
                    ad["put"](r.record_id, ad["load"](json.loads(raw)))
                    self._hashes[(store, r.record_id)] = hashlib.sha256(raw.encode()).hexdigest()
                    n += 1
                except Exception as exc:                          # a bad record must never stop the service from starting
                    skipped += 1
                    log.error("engine_state: skipped unreadable %s/%s (%s)", store, r.record_id, type(exc).__name__)
            counts[store] = n
        for r in by_store.get("patient_ecg", []):
            try:
                d = json.loads(decrypt_str(r.payload)); rec = self.o.patient_registry.patients.get(r.record_id)
                if rec is not None:
                    wave = np.frombuffer(base64.b64decode(d["b64"]), dtype="<f4").reshape(d["shape"]).copy()
                    rec.latest_ecg = {"wave": wave, "sample_rate_hz": d["sample_rate_hz"], "source": d["source"], "stored_at": d["stored_at"]}
                    self._ecg_stamp[r.record_id] = d["stored_at"]
            except Exception as exc:
                skipped += 1
                log.error("engine_state: skipped unreadable ECG for %s (%s)", r.record_id, type(exc).__name__)
        counts["skipped_unreadable"] = skipped
        self.loaded = counts
        return counts

    # ---------------- save ----------------
    def _collect(self) -> tuple[list[tuple[str, str, str]], set[tuple[str, str]]]:
        changed, present = [], set()
        for store, ad in self.adapters.items():
            for rid, obj in list(ad["items"]().items()):
                try:
                    raw = json.dumps(ad["dump"](obj), default=_json_default, sort_keys=True)
                except Exception as exc:
                    log.error("engine_state: could not serialise %s/%s (%s)", store, rid, type(exc).__name__)
                    present.add((store, rid)); continue                # keep whatever was stored last time
                present.add((store, rid))
                h = hashlib.sha256(raw.encode()).hexdigest()
                if self._hashes.get((store, rid)) != h:
                    changed.append((store, rid, raw)); self._hashes[(store, rid)] = h
        for pid, rec in list(self.o.patient_registry.patients.items()):
            ecg = rec.latest_ecg
            if ecg and self._ecg_stamp.get(pid) != ecg["stored_at"]:
                w = np.ascontiguousarray(ecg["wave"], dtype="<f4")
                raw = json.dumps({"sample_rate_hz": ecg["sample_rate_hz"], "source": ecg["source"], "stored_at": ecg["stored_at"],
                                  "shape": list(w.shape), "b64": base64.b64encode(w.tobytes()).decode()})
                changed.append(("patient_ecg", pid, raw)); self._ecg_stamp[pid] = ecg["stored_at"]
            if ecg:
                present.add(("patient_ecg", pid))
        return changed, present

    def _prepare(self):
        """Serialise on the caller's thread (the event loop), so no record is read while a request is changing it."""
        changed, present = self._collect()
        gone = [k for k in list(self._hashes) if k not in present]
        gone_ecg = [("patient_ecg", pid) for pid in list(self._ecg_stamp) if ("patient_ecg", pid) not in present]
        for k in gone_ecg:
            self._ecg_stamp.pop(k[1], None)
        return changed, gone, gone_ecg

    def _write(self, prepared) -> dict[str, int]:
        changed, gone, gone_ecg = prepared
        try:
            with self.session_factory() as db:
                for store, rid, raw in changed:
                    db.merge(EngineRecord(store=store, record_id=rid, payload=encrypt_str(raw), updated_at=datetime.now(timezone.utc)))
                for store, rid in gone + gone_ecg:
                    db.query(EngineRecord).filter(EngineRecord.store == store, EngineRecord.record_id == rid).delete()
                db.commit()
            for k in gone:
                self._hashes.pop(k, None)
            self.last_saved_at, self.last_error = time.time(), None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}"
            log.error("engine_state: save failed (%s); will retry", type(exc).__name__)
            for store, rid, raw in changed:                          # force a retry next cycle
                self._hashes.pop((store, rid), None)
                if store == "patient_ecg":
                    self._ecg_stamp.pop(rid, None)
            return {"written": 0, "deleted": 0}
        return {"written": len(changed), "deleted": len(gone) + len(gone_ecg)}

    def save_now(self) -> dict[str, int]:
        return self._write(self._prepare())

    async def run_forever(self):
        while True:
            await asyncio.sleep(self.interval_s)
            try:
                prepared = self._prepare()
                await asyncio.to_thread(self._write, prepared)
            except Exception as exc:                                  # never let the saver die
                log.error("engine_state: saver error (%s)", type(exc).__name__)

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "last_saved_at": self.last_saved_at, "last_error": self.last_error, "loaded_at_startup": self.loaded,
                "save_interval_s": self.interval_s}
