"""Password security helpers for ProFace.

Centralises password hashing so auth logic is testable without importing
the heavy CV stack (dlib/OpenCV). Uses Werkzeug's scrypt-based hashes.

Legacy support: passwords created before hashing was introduced are stored
in plaintext. verify_password() accepts those once, and the caller is
expected to re-hash them immediately (see app.py login routes).
"""
from werkzeug.security import check_password_hash, generate_password_hash

_HASH_PREFIXES = ("scrypt:", "pbkdf2:")


def hash_password(plain: str) -> str:
    """Hash a plaintext password for storage. Never store the plaintext."""
    return generate_password_hash(plain)


def is_hashed(value: str) -> bool:
    """True if the stored value already looks like a password hash."""
    return isinstance(value, str) and value.startswith(_HASH_PREFIXES)


def verify_password(stored: str, plain: str) -> bool:
    """Check a login attempt against the stored value.

    Accepts modern hashes via Werkzeug, plus legacy plaintext values so
    existing accounts keep working until they are transparently upgraded
    to hashes on their next login.
    """
    if not stored or not plain:
        return False
    if is_hashed(stored):
        return check_password_hash(stored, plain)
    # Legacy plaintext comparison (constant-time not critical here since
    # the value is upgraded to a hash immediately after a successful login).
    return stored == plain
