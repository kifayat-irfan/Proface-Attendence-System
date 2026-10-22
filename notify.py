"""Absence alerts to parents via email.

SMTP is configured entirely through environment variables — no credentials
in code, ever:

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@example.com
    SMTP_PASSWORD=your-app-password

For Gmail, create an App Password (Google Account > Security), not your
real password. If SMTP is not configured, send_absence_alert() reports
that plainly instead of failing silently.
"""
import os
import smtplib
from email.message import EmailMessage


def smtp_configured():
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER"))


def send_absence_alert(parent_email, student_name, course, pct, present, total):
    """Email one parent about their child's low attendance.

    Returns (ok: bool, message: str). Never raises for config problems.
    """
    if not parent_email:
        return False, "no parent email on file"
    if not smtp_configured():
        return False, "SMTP not configured (set SMTP_HOST/SMTP_USER/SMTP_PASSWORD)"

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ.get("SMTP_PASSWORD", "")

    msg = EmailMessage()
    msg["Subject"] = f"Attendance Alert: {student_name} at {pct}% in {course}"
    msg["From"] = user
    msg["To"] = parent_email
    msg.set_content(
        f"Dear Parent,\n\n"
        f"This is an automated attendance alert from the ProFace Attendance System.\n\n"
        f"Student: {student_name}\n"
        f"Course: {course}\n"
        f"Attendance: {present}/{total} lectures ({pct}%)\n\n"
        f"Your child's attendance has fallen below the required threshold. "
        f"Please contact the course faculty for details.\n\n"
        f"— ProFace Attendance System"
    )
    try:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.starttls()
            if password:
                s.login(user, password)
            s.send_message(msg)
    except Exception as e:  # network/auth failures -> report, don't crash
        return False, f"send failed: {e}"
    return True, "sent"
