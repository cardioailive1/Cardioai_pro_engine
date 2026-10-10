"""
CardioAI Pro — PACS C-FIND/C-MOVE, tested against a real local SCP
========================================================================
Stands up two real DICOM network services on localhost using pynetdicom's
own server support — a Query/Retrieve SCP (answers C-FIND, and C-MOVE by
pushing matched instances to a destination AE) and a Storage SCP (the
C-MOVE destination, receives the pushed instances via a real C-STORE) —
then drives DICOMService.pacs_find() / pacs_move() (integrations/dicom.py)
against them exactly as a caller would against a real hospital PACS.

This is NOT mocking pynetdicom or the DICOM protocol: every association,
C-FIND, C-MOVE, and C-STORE below is a real DIMSE exchange over a real
TCP socket on localhost. What's missing, same limitation already
disclosed for integrations/iomt_client.py's local-mock testing, is a
round-trip against an actual hospital PACS — this project has no
credentials or network access to one.

Run directly:

    python -m integrations.pacs_local_test
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import SecondaryCaptureImageStorage, ExplicitVRLittleEndian, generate_uid
from pynetdicom import AE, evt, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove,
)
from pynetdicom.presentation import build_context

from integrations.dicom import DICOMService

FIND_MOVE_PORT = 11112
STORAGE_PORT = 11113
STORAGE_AE_TITLE = "CARDIOAI_STORE"
QR_AE_TITLE = "TEST_PACS"

FAKE_PATIENT_ID = "P-1001"
FAKE_STUDY_UID = "1.2.826.0.1.3680043.8.498.99999999999999999999999999999991"


def _make_fake_instance() -> Dataset:
    """A minimal but structurally valid Secondary Capture dataset — enough for a real C-STORE, not a real image."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = FAKE_PATIENT_ID
    ds.PatientName = "Test^Patient"
    ds.StudyInstanceUID = FAKE_STUDY_UID
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = "ACC001"
    ds.Modality = "US"
    ds.ModalitiesInStudy = "US"
    ds.NumberOfStudyRelatedInstances = "1"
    ds.Rows = 2
    ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = bytes([0, 64, 128, 255])
    return ds


def handle_find(event):
    identifier = event.identifier
    if getattr(identifier, "PatientID", None) not in (None, "", FAKE_PATIENT_ID):
        return  # no matches — generator exhausted, pynetdicom sends final 0x0000 automatically
    response = Dataset()
    response.StudyInstanceUID = FAKE_STUDY_UID
    response.PatientID = FAKE_PATIENT_ID
    response.StudyDate = "20260101"
    response.StudyTime = "120000"
    response.ModalitiesInStudy = "US"
    response.NumberOfStudyRelatedInstances = "1"
    response.AccessionNumber = "ACC001"
    yield 0xFF00, response


def handle_move(event):
    destinations = {STORAGE_AE_TITLE: ("127.0.0.1", STORAGE_PORT)}
    try:
        addr, port = destinations[event.move_destination]
    except KeyError:
        yield None, None
        return

    instance = _make_fake_instance()
    contexts = [build_context(SecondaryCaptureImageStorage)]
    yield addr, port, {"contexts": contexts}
    yield 1  # number of sub-operations
    yield 0xFF00, instance


received_store = []


def handle_store(event):
    ds = event.dataset
    ds.file_meta = event.file_meta
    received_store.append(str(ds.SOPInstanceUID))
    return 0x0000


def main() -> None:
    qr_ae = AE(ae_title=QR_AE_TITLE)
    qr_ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)
    qr_ae.add_supported_context(PatientRootQueryRetrieveInformationModelMove)
    # The QR SCP also acts as a C-STORE SCU during C-MOVE (to push instances
    # to the destination), so it needs this as a REQUESTED context too.
    qr_ae.add_requested_context(SecondaryCaptureImageStorage, ALL_TRANSFER_SYNTAXES)

    store_ae = AE(ae_title=STORAGE_AE_TITLE)
    store_ae.add_supported_context(SecondaryCaptureImageStorage, ALL_TRANSFER_SYNTAXES)

    qr_server = qr_ae.start_server(
        ("127.0.0.1", FIND_MOVE_PORT), block=False,
        evt_handlers=[(evt.EVT_C_FIND, handle_find), (evt.EVT_C_MOVE, handle_move)],
    )
    store_server = store_ae.start_server(
        ("127.0.0.1", STORAGE_PORT), block=False,
        evt_handlers=[(evt.EVT_C_STORE, handle_store)],
    )
    time.sleep(0.3)  # let both listeners come up before the SCU connects

    try:
        svc = DICOMService()

        print("--- C-FIND: matching patient ---")
        results = svc.pacs_find(FAKE_PATIENT_ID, QR_AE_TITLE, "127.0.0.1", FIND_MOVE_PORT)
        print(results)
        assert len(results) == 1, f"expected 1 match, got {len(results)}"
        assert results[0]["study_instance_uid"] == FAKE_STUDY_UID
        assert results[0]["accession_number"] == "ACC001"
        print("C-FIND (match) OK")

        print("--- C-FIND: non-matching patient ---")
        no_results = svc.pacs_find("NOBODY-HERE", QR_AE_TITLE, "127.0.0.1", FIND_MOVE_PORT)
        assert no_results == [], f"expected no matches, got {no_results}"
        print("C-FIND (no match) OK")

        print("--- C-MOVE ---")
        move_result = svc.pacs_move(FAKE_STUDY_UID, STORAGE_AE_TITLE, "127.0.0.1", FIND_MOVE_PORT, QR_AE_TITLE)
        print(move_result)
        assert move_result["completed_suboperations"] == 1, move_result
        assert move_result["failed_suboperations"] == 0, move_result
        time.sleep(0.2)  # the Storage SCP's handler runs on its own thread
        assert len(received_store) == 1, f"Storage SCP never received the C-STORE: {received_store}"
        print(f"C-MOVE OK — Storage SCP actually received SOP Instance {received_store[0]}")

        print("--- C-MOVE: unknown destination AE title ---")
        try:
            svc.pacs_move(FAKE_STUDY_UID, "NOT_CONFIGURED", "127.0.0.1", FIND_MOVE_PORT, QR_AE_TITLE)
            print("FAIL: expected an association/DIMSE failure for an unknown destination")
            sys.exit(1)
        except RuntimeError as exc:
            print(f"Correctly rejected unknown destination: {exc}")

        print("\nALL PACS C-FIND/C-MOVE CHECKS PASSED (real DICOM protocol, local SCP)")
    finally:
        qr_server.shutdown()
        store_server.shutdown()


if __name__ == "__main__":
    main()
