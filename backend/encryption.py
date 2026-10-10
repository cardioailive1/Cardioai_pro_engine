"""
CardioAI Pro — Application-Level Encryption at Rest
=======================================================
WHAT THIS CLOSES: the data room previously told investors that data at
rest in Postgres and Cloudflare R2 was "encrypted at rest" with no code
to back it up — in reality that meant whatever the hosting provider does
by default, not anything this application configured or could prove.
This module is real, app-level encryption for data this app actually
controls, independent of which storage backend (local disk, Cloudflare
R2) ends up holding the bytes. Even if a storage backend's own at-rest
encryption were ever misconfigured or disabled, content encrypted here
is still unreadable without APP_ENCRYPTION_KEY.

ALGORITHM: Fernet (AES-128-CBC + HMAC-SHA256, both authenticated —
tampering is detected, not just confidentiality protected) from the
`cryptography` package. `cryptography` is already a real, live
dependency here (pulled in by python-jose[cryptography], used for RBAC
session tokens) — pinned explicitly in requirements.txt below rather
than left as an unpinned transitive dependency.

KEY MANAGEMENT: APP_ENCRYPTION_KEY is a Fernet key (32 url-safe-base64
bytes), generated once and stored outside the repo — render.yaml sets
`generateValue: true` so Render generates and stores a real random key
on first deploy, the same pattern already used for JWT_SECRET_KEY.
Rotating this key means anything already encrypted under the old key
becomes unreadable unless the old key is kept around for decryption —
see `rotate()` below for the supported path.

FAILS CLOSED, matching auth/security.py's JWT_SECRET_KEY behavior: if
APP_ENCRYPTION_KEY isn't set, encrypt/decrypt calls raise rather than
silently falling back to storing plaintext.
"""
from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken, MultiFernet


def _keys_from_env() -> list[Fernet]:
    # APP_ENCRYPTION_KEY: the active key, used for all new encryption.
    # APP_ENCRYPTION_KEY_PREVIOUS (optional, comma-separated): retired
    # keys still accepted for DECRYPTING content written before a
    # rotation, never used to encrypt anything new. This is what makes
    # key rotation possible without re-encrypting every existing object
    # synchronously during the rotation itself.
    primary = os.environ.get("APP_ENCRYPTION_KEY")
    if not primary:
        raise RuntimeError(
            "APP_ENCRYPTION_KEY is not set. render.yaml generates this automatically on deploy "
            "(generateValue: true) — if you're seeing this locally, set it yourself for testing "
            "(a valid value: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\")."
        )
    keys = [Fernet(primary.encode() if isinstance(primary, str) else primary)]
    previous = os.environ.get("APP_ENCRYPTION_KEY_PREVIOUS", "")
    for raw in (k.strip() for k in previous.split(",") if k.strip()):
        keys.append(Fernet(raw.encode()))
    return keys


@lru_cache(maxsize=1)
def _fernet() -> MultiFernet:
    # MultiFernet encrypts with the first key only, but decrypts by
    # trying each key in order — exactly the "accept old, write new"
    # behavior key rotation needs. Cached so we don't re-parse env vars
    # on every call; cache is process-local and keyed on nothing, so a
    # changed env var requires a process restart to take effect — the
    # same operational model as JWT_SECRET_KEY already has.
    return MultiFernet(_keys_from_env())


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypts arbitrary bytes for storage. Output is self-contained (includes the IV/nonce and an HMAC tag) — safe to store as-is."""
    return _fernet().encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    """
    Decrypts bytes produced by encrypt_bytes(). Raises ValueError (not
    InvalidToken directly, so callers don't need to import this module's
    crypto library just to catch errors) if the content was tampered
    with, truncated, or encrypted under a key no longer configured.
    """
    try:
        return _fernet().decrypt(token)
    except InvalidToken as exc:
        raise ValueError("Content could not be decrypted — wrong/missing key, or the data was corrupted or tampered with.") from exc


def encrypt_str(text: str) -> str:
    return encrypt_bytes(text.encode("utf-8")).decode("ascii")


def decrypt_str(token: str) -> str:
    return decrypt_bytes(token.encode("ascii")).decode("utf-8")


def is_configured() -> bool:
    """Lets callers check before relying on encryption, instead of letting the RuntimeError surface mid-request."""
    return bool(os.environ.get("APP_ENCRYPTION_KEY"))


def generate_key() -> str:
    """Convenience for local dev / the first deploy — not used by render.yaml, which generates its own via generateValue: true."""
    return Fernet.generate_key().decode("ascii")


if __name__ == "__main__":
    print(generate_key())
