"""Unit tests for ProFace password security (security.py).

Run:  python -m pytest tests/ -v
Needs only Werkzeug (no dlib/OpenCV), so these run anywhere.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import hash_password, verify_password, is_hashed


def test_hash_is_not_plaintext():
    h = hash_password("secret123")
    assert h != "secret123"
    assert is_hashed(h)


def test_verify_correct_password():
    h = hash_password("secret123")
    assert verify_password(h, "secret123") is True


def test_verify_wrong_password():
    h = hash_password("secret123")
    assert verify_password(h, "wrong") is False


def test_legacy_plaintext_still_verifies_once():
    # Pre-hashing accounts keep working until transparent upgrade on login
    assert verify_password("oldplain", "oldplain") is True
    assert verify_password("oldplain", "nope") is False
    assert is_hashed("oldplain") is False


def test_empty_values_rejected():
    h = hash_password("x")
    assert verify_password(h, "") is False
    assert verify_password("", "x") is False
    assert verify_password(None, "x") is False


def test_two_hashes_differ():
    # Salting: same password -> different hashes, both verify
    assert hash_password("same") != hash_password("same")
    assert verify_password(hash_password("same"), "same") is True


# --- Liveness trust policy ---
from security import is_live_face, LIVENESS_THRESHOLD


def test_cnn_alone_can_pass_liveness():
    assert is_live_face(False, 0.95) is True


def test_blink_alone_can_pass_liveness():
    assert is_live_face(True, 0.10) is True


def test_both_failing_rejects():
    assert is_live_face(False, 0.10) is False


def test_threshold_boundary():
    assert is_live_face(False, LIVENESS_THRESHOLD) is True
    assert is_live_face(False, LIVENESS_THRESHOLD - 0.01) is False
