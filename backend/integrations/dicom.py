"""
CardioAI Pro — PACS & DICOM Integration
===========================================
Two responsibilities:

  1. DICOM metadata extraction from an uploaded study (via pydicom) —
     used when a cardiologist's workstation or PACS forwards a study for
     AI-assisted review.
  2. PACS query/retrieve (DICOM C-FIND/C-MOVE) over the real DICOM
     network protocol, via `pynetdicom` — a real SCU (Service Class
     User): it associates with a PACS, sends real C-FIND/C-MOVE
     messages, and parses the real DIMSE status codes in the response.

WHAT'S REAL HERE AND WHAT ISN'T: the C-FIND/C-MOVE code below is not a
stub — it is tested end to end against a real DICOM Query/Retrieve SCP
(Service Class Provider), using pynetdicom's own server support to run
one locally — see integrations/pacs_local_test.py, which stands up a
real QR SCP plus a real Storage SCP on localhost, no mocking of the
DICOM protocol itself. What ISN'T tested is a round-trip against an
actual hospital PACS, because this project has no credentials or
network access to one — exactly the same honest limitation
integrations/iomt_client.py already discloses for its own local-mock
testing. Point `pacs_host`/`pacs_port`/`ae_title` at a real PACS and
this code should work against it unchanged; nothing about the
association, C-FIND identifier, or C-MOVE handling is hospital-specific.
"""
from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np


class DICOMService:
    def summarize(self, record: dict[str, Any]) -> dict[str, Any]:
        """Summarizes a record already carrying DICOM-ish fields (used by the
        ingestion pipeline when the caller has pre-extracted tags)."""
        return {
            "modality": record.get("modality_tag", "US"),  # e.g. "US" echocardiogram, "XA" angiography
            "study_instance_uid": record.get("study_instance_uid"),
            "patient_id": record.get("patient_id"),
            "study_date": record.get("study_date"),
            "series_count": record.get("series_count", 1),
            "source": "PACS forward (simulated)",
        }

    def extract_metadata_from_bytes(self, file_bytes: bytes) -> dict[str, Any]:
        """Extracts key tags from an actual DICOM file using pydicom."""
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pydicom is not installed — see requirements.txt") from exc

        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        return {
            "patient_id": str(getattr(ds, "PatientID", "unknown")),
            "modality": str(getattr(ds, "Modality", "unknown")),
            "study_date": str(getattr(ds, "StudyDate", "unknown")),
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "unknown")),
            "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "unknown")),
            "manufacturer": str(getattr(ds, "Manufacturer", "unknown")),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "bits_allocated": getattr(ds, "BitsAllocated", None),
        }

    def extract_pixel_features(self, file_bytes: bytes) -> dict[str, Any]:
        """
        Extracts REAL quantitative features from the actual pixel data — not
        just header metadata. This is genuine image analysis (intensity
        statistics, histogram entropy, a simple edge-density texture proxy),
        but it is explicitly NOT a diagnostic interpretation: nothing here
        says what these numbers mean clinically. That gap is intentional —
        see inference/models.py's predict_imaging(), which reports these
        features as "extracted" while honestly marking overall status as
        awaiting_trained_model. A real diagnostic model would take these
        (or richer, learned) features as input; this function stops at
        producing them.
        """
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pydicom is not installed — see requirements.txt") from exc

        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        try:
            pixels = ds.pixel_array.astype(np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read pixel data (transfer syntax may need an additional pydicom plugin): {exc}"
            ) from exc

        # Use the middle frame for multi-frame series (e.g. cine loops) so a
        # single representative frame drives the statistics below.
        if pixels.ndim == 3:
            pixels = pixels[pixels.shape[0] // 2]

        flat = pixels.ravel()
        hist, _ = np.histogram(flat, bins=64, density=True)
        hist = hist[hist > 0]
        entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0

        # Simple gradient-magnitude texture proxy — real computation on real
        # pixels, not a placeholder. High values suggest a lot of edge/
        # structural detail (e.g. vessel borders); this is a coarse texture
        # signal, not a segmentation or lesion-detection result.
        gy, gx = np.gradient(pixels)
        gradient_magnitude = float(np.sqrt(gx**2 + gy**2).mean())

        return {
            "shape": list(pixels.shape),
            "intensity_mean": round(float(pixels.mean()), 3),
            "intensity_std": round(float(pixels.std()), 3),
            "intensity_min": round(float(pixels.min()), 3),
            "intensity_max": round(float(pixels.max()), 3),
            "histogram_entropy": round(entropy, 4),
            "gradient_magnitude_mean": round(gradient_magnitude, 4),
            "note": "Real features extracted from actual pixel data — not a diagnostic interpretation. No trained model exists to say what these values mean clinically.",
        }

    # ---- PACS query/retrieve (C-FIND / C-MOVE), real DICOM network protocol --
    def pacs_find(
        self, patient_id: str, ae_title: str, pacs_host: str, pacs_port: int,
        calling_ae_title: str = "CARDIOAI_PRO", query_level: str = "STUDY", timeout: float = 15.0,
    ) -> list[dict[str, Any]]:
        """
        Real DICOM C-FIND (Patient Root, Study level by default) against a
        PACS: associates, sends the query, and collects every pending match
        — not a mock of the protocol, an actual pynetdicom AE association.
        See integrations/pacs_local_test.py for this running against a real
        local SCP.

        Raises RuntimeError on association failure, a timeout/abort mid-query,
        or a non-success DIMSE status — callers (the /api/dicom/pacs/find
        route) turn that into a 502/504 rather than a silent empty result,
        so a genuine PACS-side failure is never indistinguishable from "no
        matching studies."
        """
        from pynetdicom import AE
        from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind
        from pydicom.dataset import Dataset

        ae = AE(ae_title=calling_ae_title)
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        ae.network_timeout = timeout
        ae.acse_timeout = timeout
        ae.dimse_timeout = timeout

        identifier = Dataset()
        identifier.QueryRetrieveLevel = query_level
        identifier.PatientID = patient_id
        # Empty-string return keys — present but unspecified, so the SCP
        # knows to populate them in the response instead of using them to
        # filter the query (DICOM's standard "universal matching" idiom).
        identifier.StudyInstanceUID = ""
        identifier.StudyDate = ""
        identifier.StudyTime = ""
        identifier.ModalitiesInStudy = ""
        identifier.NumberOfStudyRelatedInstances = ""
        identifier.AccessionNumber = ""

        assoc = ae.associate(pacs_host, pacs_port, ae_title=ae_title)
        if not assoc.is_established:
            raise RuntimeError(f"Could not associate with PACS at {pacs_host}:{pacs_port} (AE title {ae_title!r}).")

        results: list[dict[str, Any]] = []
        try:
            responses = assoc.send_c_find(identifier, PatientRootQueryRetrieveInformationModelFind)
            for status, rsp_identifier in responses:
                if status is None:
                    raise RuntimeError("C-FIND connection timed out, was aborted, or returned an invalid response.")
                if status.Status in (0xFF00, 0xFF01):  # Pending — this is a match, more may follow
                    if rsp_identifier is not None:
                        results.append({
                            "study_instance_uid": str(getattr(rsp_identifier, "StudyInstanceUID", "")),
                            "patient_id": str(getattr(rsp_identifier, "PatientID", "")),
                            "study_date": str(getattr(rsp_identifier, "StudyDate", "")),
                            "study_time": str(getattr(rsp_identifier, "StudyTime", "")),
                            "modalities_in_study": str(getattr(rsp_identifier, "ModalitiesInStudy", "")),
                            "number_of_instances": str(getattr(rsp_identifier, "NumberOfStudyRelatedInstances", "")),
                            "accession_number": str(getattr(rsp_identifier, "AccessionNumber", "")),
                        })
                elif status.Status == 0x0000:
                    pass  # Success — query complete, no further matches
                else:
                    raise RuntimeError(f"PACS refused or failed the C-FIND query — DIMSE status 0x{status.Status:04X}.")
        finally:
            assoc.release()

        return results

    def pacs_move(
        self, study_instance_uid: str, destination_ae_title: str, pacs_host: str, pacs_port: int,
        pacs_ae_title: str, calling_ae_title: str = "CARDIOAI_PRO", timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Real DICOM C-MOVE: asks the PACS to push a study's instances to
        `destination_ae_title` — a separate, already-known Storage SCP
        (which is NOT this call; C-MOVE is "tell the PACS to send it
        somewhere," not a direct file transfer back to the caller). The
        PACS must already have that destination AE title configured/
        allow-listed, same as any real DICOM network — this call cannot
        invent that association on the PACS's side.

        Returns the final sub-operation counts (completed/failed/warning/
        remaining) from the last DIMSE status received, which is how a
        caller tells "all instances moved" from "some failed" from "the
        destination AE title wasn't recognized by the PACS" (0xA801).
        """
        from pynetdicom import AE
        from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove
        from pydicom.dataset import Dataset

        ae = AE(ae_title=calling_ae_title)
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)
        ae.network_timeout = timeout
        ae.acse_timeout = timeout
        ae.dimse_timeout = timeout

        identifier = Dataset()
        identifier.QueryRetrieveLevel = "STUDY"
        identifier.StudyInstanceUID = study_instance_uid

        assoc = ae.associate(pacs_host, pacs_port, ae_title=pacs_ae_title)
        if not assoc.is_established:
            raise RuntimeError(f"Could not associate with PACS at {pacs_host}:{pacs_port} (AE title {pacs_ae_title!r}).")

        completed = failed = warning = remaining = 0
        final_status: Optional[int] = None
        try:
            responses = assoc.send_c_move(identifier, destination_ae_title, PatientRootQueryRetrieveInformationModelMove)
            for status, _rsp_identifier in responses:
                if status is None:
                    raise RuntimeError("C-MOVE connection timed out, was aborted, or returned an invalid response.")
                final_status = status.Status
                completed = getattr(status, "NumberOfCompletedSuboperations", completed)
                failed = getattr(status, "NumberOfFailedSuboperations", failed)
                warning = getattr(status, "NumberOfWarningSuboperations", warning)
                remaining = getattr(status, "NumberOfRemainingSuboperations", remaining)
        finally:
            assoc.release()

        if final_status not in (0x0000, 0xFF00, None):
            raise RuntimeError(
                f"PACS refused or failed the C-MOVE request — DIMSE status "
                f"0x{final_status:04X} (completed={completed}, failed={failed}, warning={warning})."
            )

        return {
            "status": f"0x{final_status:04X}" if final_status is not None else None,
            "completed_suboperations": completed,
            "failed_suboperations": failed,
            "warning_suboperations": warning,
            "remaining_suboperations": remaining,
            "destination_ae_title": destination_ae_title,
        }
