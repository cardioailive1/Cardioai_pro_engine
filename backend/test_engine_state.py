"""Persistence of the engine's working data: round trip, change detection, deletion, bad records, encryption at rest."""
import asyncio
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from billing.contracts import ContractRegistry
from billing.enrollment import EnrolledMember, EnrollmentRegistry
from care_management.tasks import CareTaskRegistry
from engine_state import EngineRecord, StatePersister
from orchestrator.care_continuum import CareContinuumTracker
from orchestrator.patient_registry import PatientRegistry
from reports.population_report import PopulationAggregator


def make_orch():
    return SimpleNamespace(patient_registry=PatientRegistry(), continuum_tracker=CareContinuumTracker(), population_aggregator=PopulationAggregator(),
                           care_task_registry=CareTaskRegistry(), enrollment_registry=EnrollmentRegistry(), contract_registry=ContractRegistry())


@pytest.fixture
def session_factory():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=__import__("sqlalchemy.pool", fromlist=["StaticPool"]).StaticPool)
    EngineRecord.__table__.create(eng)
    return sessionmaker(bind=eng)


def populate(o):
    r = o.patient_registry.admit_patient(patient_id="MRN-P1", name="Jane Q Patient", age=71, sex="F", room="4A", cardiologist="Dr X", nurse="N Y", allergies=["penicillin"],
                                         medications=["metoprolol"], payer="Cigna")[0]
    o.patient_registry.store_ecg("MRN-P1", np.random.RandomState(0).randn(12, 600), 250, "MUSE feed")
    r.latest_mace = {"status": "ok", "prediction": {"score": np.float32(0.42), "risk_tier": "moderate"}, "message": "m", "source": {}, "scored_at": 1.0}
    o.patient_registry.add_order("MRN-P1", "ECG waveform MACE analysis", "Dr X", "completed", "MACE 0.42")
    o.continuum_tracker.record_diagnostic_finding("MRN-P1", "Atrial fibrillation"); o.continuum_tracker.mark_treatment("MRN-P1", "Signed off by Dr X")
    o.population_aggregator.record("M1", 66, 2, 0.4, "moderate", claim_id="C1", total_charge_amount=1200.0, diagnosis_codes=["I10"])
    o.care_task_registry.create_task("M1", "outreach_call", "Nurse A", "call")
    o.enrollment_registry.members["M1"] = EnrolledMember("M1", "A", "B", "1960-01-01", "F", "active", "2026-01-01")
    c = o.contract_registry.create_contract("Cigna", 12.5, 1000, date(2026, 1, 1), date(2026, 12, 31))
    o.contract_registry.invoices.append(__import__("billing.contracts", fromlist=["Invoice"]).Invoice("INV-1", c.contract_id, "2026-09", "contracted", 1000, 12.5, 1.0, 12500.0))


def test_round_trip_survives_a_restart(session_factory):
    a = make_orch(); populate(a)
    p = StatePersister(a, session_factory, interval_s=1)
    assert p.save_now()["written"] >= 8
    b = make_orch(); q = StatePersister(b, session_factory); counts = q.load()
    assert counts["patients"] == 1 and counts["continuum"] == 1 and counts["payer_members"] == 1 and counts["contracts"] == 1 and counts["invoices"] == 1 and counts["skipped_unreadable"] == 0
    pa, pb = a.patient_registry.patients["MRN-P1"], b.patient_registry.patients["MRN-P1"]
    assert pb.name == "Jane Q Patient" and pb.allergies == ["penicillin"] and pb.orders[0]["order_type"] == "ECG waveform MACE analysis"
    assert pb.latest_mace["prediction"]["score"] == pytest.approx(0.42, abs=1e-6)
    assert pb.latest_ecg["source"] == "MUSE feed" and np.allclose(pb.latest_ecg["wave"], pa.latest_ecg["wave"].astype("<f4")) and pb.latest_ecg["sample_rate_hz"] == 250
    cb = b.continuum_tracker.patients["MRN-P1"]
    assert cb.stage.value == "treatment" and cb.history[-1]["reason"] == "Signed off by Dr X"
    assert b.population_aggregator.members["M1"].claim_history[0]["charge_amount"] == 1200.0
    assert list(b.care_task_registry.tasks.values())[0].status == list(a.care_task_registry.tasks.values())[0].status
    assert list(b.care_task_registry.tasks.values())[0].member_id == "M1" and b.enrollment_registry.members["M1"].dob == "1960-01-01"
    ctr = list(b.contract_registry.contracts.values())[0]
    assert ctr.start_date == date(2026, 1, 1) and ctr.end_date == date(2026, 12, 31) and b.contract_registry.invoices[0].amount == 12500.0
    assert q.save_now() == {"written": 0, "deleted": 0}                      # reloaded state is already saved


def test_only_changed_records_are_written_and_removed_ones_deleted(session_factory):
    o = make_orch(); populate(o); p = StatePersister(o, session_factory)
    p.save_now()
    assert p.save_now() == {"written": 0, "deleted": 0}
    o.patient_registry.patients["MRN-P1"].phone = "555-0100"
    assert p.save_now()["written"] == 1
    del o.population_aggregator.members["M1"]
    o.patient_registry.patients["MRN-P1"].latest_ecg = None
    r = p.save_now(); assert r["deleted"] == 2
    with session_factory() as db:
        assert db.query(EngineRecord).filter_by(store="payer_members").count() == 0 and db.query(EngineRecord).filter_by(store="patient_ecg").count() == 0


def test_database_holds_no_readable_patient_data(session_factory):
    o = make_orch(); populate(o); StatePersister(o, session_factory).save_now()
    with session_factory() as db:
        blob = " ".join(r.payload for r in db.query(EngineRecord).all())
    assert blob and "Jane" not in blob and "penicillin" not in blob and "Cigna" not in blob and "MRN-P1" not in blob


def test_one_unreadable_record_does_not_stop_loading(session_factory):
    o = make_orch(); populate(o); StatePersister(o, session_factory).save_now()
    with session_factory() as db:
        db.add(EngineRecord(store="patients", record_id="BAD", payload="not-encrypted")); db.commit()
    b = make_orch(); counts = StatePersister(b, session_factory).load()
    assert counts["patients"] == 1 and counts["skipped_unreadable"] == 1 and "BAD" not in b.patient_registry.patients


def test_failed_save_is_retried_next_time(session_factory, monkeypatch):
    o = make_orch(); populate(o); p = StatePersister(o, session_factory)
    good = p.session_factory
    p.session_factory = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    assert p.save_now() == {"written": 0, "deleted": 0} and p.last_error == "RuntimeError"
    p.session_factory = good
    assert p.save_now()["written"] >= 8 and p.last_error is None


def test_background_saver_runs(session_factory):
    o = make_orch(); populate(o); p = StatePersister(o, session_factory, interval_s=0.05)
    async def go():
        t = asyncio.create_task(p.run_forever()); await asyncio.sleep(0.4); t.cancel()
    asyncio.run(go())
    with session_factory() as db:
        assert db.query(EngineRecord).count() >= 8
    assert p.status()["last_saved_at"] is not None
