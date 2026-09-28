"""
CardioAI Pro — Data Room Document Storage
================================================
Real upload/list/download/delete for due-diligence documents (articles
of incorporation, bylaws, employee contracts, cap table, IP filings, and
so on) — closes a real gap in the investor data room, which until now
only had hardcoded content pulled from the financial model, no actual
document storage.

TWO BACKENDS, CHOSEN AUTOMATICALLY BY WHAT'S CONFIGURED:
- Local disk (the original implementation): files live only as long as
  the current process. Render's filesystem is ephemeral — wiped on every
  redeploy, not just a crash — so this does NOT survive a redeploy or a
  free-tier spin-down. Used only when no R2 credentials are set, so the
  app still works out of the box with no setup.
- Cloudflare R2 (S3-compatible object storage): genuinely persistent.
  Used automatically once R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
  R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME are all set as environment
  variables — no code change needed to switch over.

WHY METADATA LIVES IN R2 TOO, NOT JUST THE FILE BYTES: a naive swap
would store file content in R2 but keep the document list (filename,
category, description) in the same in-memory dict as before — solving
half the problem and silently losing the other half on the next
redeploy (files would persist, but nothing would know they exist).
Metadata is stored as real S3 object metadata on each upload, and
list_all() queries R2 directly rather than an in-memory cache — a fresh
process reflects exactly what's actually in the bucket, not what it
remembers from before it restarted.

TESTED WITHOUT REAL CLOUDFLARE CREDENTIALS, STATED PLAINLY: the R2
backend is verified against `moto`'s S3 mock (real boto3 calls, real
S3-compatible API surface, no network) — this proves the integration
code is correct, but is not the same as a live round-trip against a
real Cloudflare account, which this project has no credentials for.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote, unquote

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

MAX_FILE_SIZE_BYTES = 25 * 1_000_000  # 25MB (decimal) — generous for PDFs/DOCX; guards against something oversized, not a real capacity limit either way


@dataclass
class DataRoomDocument:
    doc_id: str
    filename: str
    category: str
    description: str
    uploaded_at: float
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "filename": self.filename, "category": self.category,
            "category_label": DOCUMENT_CATEGORIES.get(self.category, self.category),
            "description": self.description, "uploaded_at": self.uploaded_at, "size_bytes": self.size_bytes,
        }


def _r2_env_configured() -> bool:
    has_endpoint = bool(os.environ.get("R2_ACCOUNT_ID") or os.environ.get("R2_ENDPOINT_URL"))
    return has_endpoint and all(os.environ.get(k) for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"))


class _LocalDiskBackend:
    """Original implementation — ephemeral, used only when R2 isn't configured."""

    def __init__(self):
        self.storage_dir = os.path.join(os.path.dirname(__file__), "_uploads")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.documents: dict[str, DataRoomDocument] = {}
        self._content: dict[str, bytes] = {}  # kept alongside self.documents; both are process-local and both vanish on restart, which is exactly the point being disclosed

    def save(self, filename: str, category: str, description: str, content: bytes) -> DataRoomDocument:
        doc_id = f"DOC-{uuid.uuid4().hex[:10]}"
        with open(os.path.join(self.storage_dir, doc_id), "wb") as f:
            f.write(content)
        doc = DataRoomDocument(doc_id=doc_id, filename=filename, category=category, description=description, uploaded_at=time.time(), size_bytes=len(content))
        self.documents[doc_id] = doc
        return doc

    def get(self, doc_id: str) -> Optional[DataRoomDocument]:
        return self.documents.get(doc_id)

    def read_content(self, doc_id: str) -> Optional[bytes]:
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        path = os.path.join(self.storage_dir, doc_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def delete(self, doc_id: str) -> bool:
        doc = self.documents.pop(doc_id, None)
        if not doc:
            return False
        path = os.path.join(self.storage_dir, doc_id)
        if os.path.exists(path):
            os.remove(path)
        return True

    def list_all(self) -> list[DataRoomDocument]:
        return list(self.documents.values())


class _R2Backend:
    """
    Cloudflare R2 via its S3-compatible API (boto3). Metadata (filename,
    category, description, uploaded_at) is stored as real S3 object
    metadata on each upload — list_all() queries R2 directly rather than
    an in-memory cache, so a fresh process reflects the bucket's actual
    contents, not stale process memory.
    """

    def __init__(self):
        import boto3
        self.bucket = os.environ["R2_BUCKET_NAME"]
        # Accept either form Cloudflare's dashboard hands you: a full
        # endpoint URL directly (R2_ENDPOINT_URL — what the current R2
        # token screen actually surfaces as "S3 API endpoint"), or just
        # the bare account ID (R2_ACCOUNT_ID) to build it from. Forcing
        # everyone to go dig the bare account ID out separately when
        # Cloudflare already handed them the full URL is friction this
        # doesn't need.
        endpoint_url = os.environ.get("R2_ENDPOINT_URL") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",  # R2's S3-compatible API expects "auto", not a real AWS region
        )

    def _doc_from_head(self, doc_id: str, head: dict[str, Any]) -> DataRoomDocument:
        # S3/R2 object metadata is ASCII-only at the HTTP-header level — a
        # real filename or description containing an em-dash, curly quote,
        # accented character, or emoji throws ParamValidationError on write
        # otherwise, which is not a hypothetical: it crashed this exact
        # startup path in production the first time a seeded description
        # used "—". URL-quoting on write / unquoting on read round-trips
        # any Unicode content safely through the ASCII-only constraint.
        meta = head.get("Metadata", {})
        return DataRoomDocument(
            doc_id=doc_id,
            filename=unquote(meta.get("filename", doc_id)),
            category=unquote(meta.get("category", "other")),
            description=unquote(meta.get("description", "")),
            uploaded_at=float(meta.get("uploaded_at", 0)),
            size_bytes=head.get("ContentLength", 0),
        )

    def save(self, filename: str, category: str, description: str, content: bytes) -> DataRoomDocument:
        doc_id = f"DOC-{uuid.uuid4().hex[:10]}"
        uploaded_at = time.time()
        self.client.put_object(
            Bucket=self.bucket, Key=doc_id, Body=content,
            Metadata={
                "filename": quote(filename), "category": quote(category),
                "description": quote(description), "uploaded_at": str(uploaded_at),
            },
        )
        return DataRoomDocument(doc_id=doc_id, filename=filename, category=category, description=description, uploaded_at=uploaded_at, size_bytes=len(content))

    def get(self, doc_id: str) -> Optional[DataRoomDocument]:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=doc_id)
        except self.client.exceptions.ClientError:
            return None
        return self._doc_from_head(doc_id, head)

    def read_content(self, doc_id: str) -> Optional[bytes]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=doc_id)
        except self.client.exceptions.ClientError:
            return None
        return obj["Body"].read()

    def delete(self, doc_id: str) -> bool:
        if self.get(doc_id) is None:
            return False
        self.client.delete_object(Bucket=self.bucket, Key=doc_id)
        return True

    def list_all(self) -> list[DataRoomDocument]:
        docs: list[DataRoomDocument] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                doc_id = obj["Key"]
                head = self.client.head_object(Bucket=self.bucket, Key=doc_id)
                docs.append(self._doc_from_head(doc_id, head))
        return docs


class DocumentRegistry:
    def __init__(self):
        if _r2_env_configured():
            self.backend = _R2Backend()
            self.storage_warning = (
                "Files stored here persist in Cloudflare R2 — they survive redeploys and spin-downs. "
                "This does not replace normal backups or access controls on the bucket itself."
            )
        else:
            self.backend = _LocalDiskBackend()
            self.storage_warning = (
                "Files stored here do NOT survive a redeploy or a free-tier spin-down "
                "— (R2_ACCOUNT_ID or R2_ENDPOINT_URL), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
                "and R2_BUCKET_NAME are not all set, so this process has no persistent storage "
                "behind it. Keep your own copy of anything uploaded here until those are configured."
            )

    def save(self, filename: str, category: str, description: str, content: bytes) -> tuple[Optional[DataRoomDocument], Optional[str]]:
        if category not in DOCUMENT_CATEGORIES:
            return None, f"category must be one of {sorted(DOCUMENT_CATEGORIES.keys())}, got {category!r}"
        if len(content) > MAX_FILE_SIZE_BYTES:
            return None, f"File is {len(content)/1_000_000:.1f}MB — the {MAX_FILE_SIZE_BYTES//1_000_000}MB limit guards against something oversized landing in storage, not a real capacity figure."
        if not filename:
            return None, "filename is required"
        return self.backend.save(filename, category, description, content), None

    def get(self, doc_id: str) -> Optional[DataRoomDocument]:
        return self.backend.get(doc_id)

    def read_content(self, doc_id: str) -> Optional[bytes]:
        return self.backend.read_content(doc_id)

    def delete(self, doc_id: str) -> bool:
        return self.backend.delete(doc_id)

    def list_all(self, category: Optional[str] = None) -> list[DataRoomDocument]:
        docs = self.backend.list_all()
        if category:
            docs = [d for d in docs if d.category == category]
        return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)

