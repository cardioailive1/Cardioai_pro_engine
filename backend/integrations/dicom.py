"""
CardioAI Pro — PACS & DICOM Integration
===========================================
Two responsibilities:

  1. DICOM metadata extraction from an uploaded study (via pydicom) —
     used when a cardiologist's workstation or PACS forwards a study for
     AI-assisted review.
  2. A PACS query/retrieve scaffold (DICOM C-FIND/C-MOVE) — stubbed here
     since establishing it requires a real PACS endpoint, AE title, and
     network access into the hospital's imaging network. `pynetdicom` is
     the library to use for the real implementation; the class below
     shows exactly where that wiring goes.
"""
from __future__ import annotations

import io
from typing import Any

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

    # ---- PACS network scaffold (C-FIND / C-MOVE) -----------------------------
    def pacs_find(self, patient_id: str, ae_title: str, pacs_host: str, pacs_port: int) -> list[dict]:
        """
        Real implementation (requires `pynetdicom` and network access to the
        hospital's PACS):

            from pynetdicom import AE
            from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind
            ae = AE(ae_title=ae_title)
            ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
            assoc = ae.associate(pacs_host, pacs_port)
            # ... build C-FIND identifier dataset with PatientID, query, collect results
            assoc.release()

        Not runnable in this environment (no PACS endpoint to connect to).
        """
        raise NotImplementedError(
            "PACS C-FIND requires network access to the hospital's PACS AE — "
            "configure pacs_host/pacs_port/ae_title against a live endpoint."
        )
