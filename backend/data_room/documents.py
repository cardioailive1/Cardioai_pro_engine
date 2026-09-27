"""
CardioAI Pro — Data Room Document Storage
================================================
Real upload/list/download/delete for due-diligence documents (articles
of incorporation, bylaws, employee contracts, cap table, IP filings, and
so on) — closes a real gap in the investor data room, which until now
only had hardcoded content pulled from the financial model, no actual
document storage.

STATED PLAINLY, NOT BURIED: this inherits the exact same limitation
every other registry in this project has — DOCUMENT_STORE and the files
written to disk both live only as long as the current process. Render's
filesystem is itself ephemeral (wiped on every redeploy, not just on a
crash), so even writing to local disk here does NOT survive a redeploy
or a free-tier spin-down, the identical failure mode that wiped the
clinical/payer data earlier in this project. This module does not
pretend otherwise — see the "storage_warning" field every endpoint
response carries, and the same disclosure surfaced directly in the
Document Room UI. Real persistence for this (and everything else still
in-memory) needs actual durable storage — a database for metadata plus
either a persistent disk or real object storage (e.g. S3) for the files
themselves — not built here.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "_uploads")
os.makedirs(STORAGE_DIR, exist_ok=True)

STORAGE_WARNING = (
    "Files stored here do NOT survive a redeploy or a free-tier spin-down "
    "— this process has no persistent disk or object storage behind it. "
    "Keep your own copy of anything uploaded here until real persistence "
    "is added."
)

DOCUMENT_CATEGORIES = {
    "corporate_legal": "Corporate & Legal",       # Articles of Incorporation, Bylaws, Cap Table, Board Resolutions
    "hr_employment": "HR & Employment",             # Employee Contracts, Offer Letters, Equity Grants
    "ip": "Intellectual Property",                  # Patent filings, Trademarks, IP Assignment Agreements
    "financial": "Financial",                        # Financial statements, Tax returns, Bank statements
    "commercial": "Commercial Agreements",           # Customer contracts, LOIs, Vendor/Partnership agreements
    "regulatory": "Regulatory & Compliance",         # FDA correspondence, HIPAA BAAs
    "insurance": "Insurance",                        # D&O, General liability
    "other": "Other",
}

MAX_FILE_SIZE_BYTES = 25 * 1_000_000  # 25MB (decimal), matching the decimal MB the error message displays — generous for PDFs/DOCX, guards against something huge landing on a small ephemeral disk


@dataclass
class DataRoomDocument:
    doc_id: str
    filename: str
    category: str
    description: str
    uploaded_at: float
    size_bytes: int
    storage_path: str  # internal — never exposed in to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "filename": self.filename, "category": self.category,
            "category_label": DOCUMENT_CATEGORIES.get(self.category, self.category),
            "description": self.description, "uploaded_at": self.uploaded_at, "size_bytes": self.size_bytes,
        }


class DocumentRegistry:
    def __init__(self):
        self.documents: dict[str, DataRoomDocument] = {}

    def save(self, filename: str, category: str, description: str, content: bytes) -> tuple[Optional[DataRoomDocument], Optional[str]]:
        if category not in DOCUMENT_CATEGORIES:
            return None, f"category must be one of {sorted(DOCUMENT_CATEGORIES.keys())}, got {category!r}"
        if len(content) > MAX_FILE_SIZE_BYTES:
            return None, f"File is {len(content)/1_000_000:.1f}MB — the {MAX_FILE_SIZE_BYTES//1_000_000}MB limit exists to guard the ephemeral disk, not as a real storage-capacity figure."
        if not filename:
            return None, "filename is required"

        doc_id = f"DOC-{uuid.uuid4().hex[:10]}"
        storage_path = os.path.join(STORAGE_DIR, doc_id)
        with open(storage_path, "wb") as f:
            f.write(content)

        doc = DataRoomDocument(
            doc_id=doc_id, filename=filename, category=category, description=description,
            uploaded_at=time.time(), size_bytes=len(content), storage_path=storage_path,
        )
        self.documents[doc_id] = doc
        return doc, None

    def get(self, doc_id: str) -> Optional[DataRoomDocument]:
        return self.documents.get(doc_id)

    def read_content(self, doc_id: str) -> Optional[bytes]:
        doc = self.documents.get(doc_id)
        if not doc or not os.path.exists(doc.storage_path):
            return None
        with open(doc.storage_path, "rb") as f:
            return f.read()

    def delete(self, doc_id: str) -> bool:
        doc = self.documents.pop(doc_id, None)
        if not doc:
            return False
        if os.path.exists(doc.storage_path):
            os.remove(doc.storage_path)
        return True

    def list_all(self, category: Optional[str] = None) -> list[DataRoomDocument]:
        docs = list(self.documents.values())
        if category:
            docs = [d for d in docs if d.category == category]
        return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)
