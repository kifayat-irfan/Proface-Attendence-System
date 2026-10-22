"""Unit tests for ProFace parent alerts (notify.py). No network use."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notify import send_absence_alert, smtp_configured


def _clear_env(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)


def test_not_configured_reports_plainly(monkeypatch):
    _clear_env(monkeypatch)
    assert smtp_configured() is False
    ok, msg = send_absence_alert("p@x.com", "Ali", "CS101", 40.0, 2, 5)
    assert ok is False
    assert "SMTP not configured" in msg


def test_missing_parent_email(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    ok, msg = send_absence_alert("", "Ali", "CS101", 40.0, 2, 5)
    assert ok is False
    assert "no parent email" in msg


def test_send_failure_reports_without_raising(monkeypatch):
    # Unreachable SMTP host -> (False, reason), never an exception
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", "9")  # discard port, nothing listens
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    ok, msg = send_absence_alert("p@x.com", "Ali", "CS101", 40.0, 2, 5)
    assert ok is False
    assert "send failed" in msg
