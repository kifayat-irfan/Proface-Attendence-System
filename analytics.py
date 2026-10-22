"""Attendance analytics for ProFace — pure SQLAlchemy, no CV dependencies.

All functions take (session, Student, Attendance, ...) so they are unit
testable against any equivalent schema without importing the Flask app.
"""
from datetime import date, timedelta

from sqlalchemy import func


def _course_lectures(session, Attendance, course):
    rows = (
        session.query(Attendance.lecture_no)
        .filter_by(course=course)
        .distinct()
        .all()
    )
    return sorted(r[0] for r in rows if r[0] is not None)


def _approved_student_count(session, Student):
    return session.query(Student).filter_by(approved=True).count()


def course_kpis(session, Student, Attendance, course):
    """Headline numbers for one course."""
    students = _approved_student_count(session, Student)
    lectures = _course_lectures(session, Attendance, course)
    n_lectures = len(lectures)
    total_marks = session.query(Attendance).filter_by(course=course).count()
    avg_pct = (
        round(100.0 * total_marks / (students * n_lectures), 1)
        if students and n_lectures
        else 0.0
    )
    return {
        "course": course,
        "students": students,
        "lectures": n_lectures,
        "total_marks": total_marks,
        "avg_attendance_pct": avg_pct,
    }


def attendance_trend(session, Student, Attendance, course, days=14):
    """Per-day % of students present over the last `days` days."""
    students = _approved_student_count(session, Student)
    since = date.today() - timedelta(days=days)
    rows = (
        session.query(
            Attendance.marked_date,
            func.count(func.distinct(Attendance.rollno)),
        )
        .filter(Attendance.course == course, Attendance.marked_date >= since)
        .group_by(Attendance.marked_date)
        .order_by(Attendance.marked_date)
        .all()
    )
    return [
        {
            "date": str(d),
            "present": p,
            "pct": round(100.0 * p / students, 1) if students else 0.0,
        }
        for d, p in rows
    ]


def defaulter_list(session, Student, Attendance, course, threshold=75.0):
    """Approved students below `threshold` % attendance, worst first."""
    lectures = _course_lectures(session, Attendance, course)
    n = len(lectures)
    if not n:
        return []
    rows = (
        session.query(
            Student.rollno,
            Student.name,
            Student.parent_email,
            func.count(Attendance.att_id),
        )
        .outerjoin(
            Attendance,
            (Attendance.rollno == Student.rollno)
            & (Attendance.course == course),
        )
        .filter(Student.approved == True)  # noqa: E712
        .group_by(Student.rollno, Student.name, Student.parent_email)
        .all()
    )
    out = []
    for rollno, name, parent_email, present in rows:
        pct = round(100.0 * present / n, 1)
        if pct < threshold:
            out.append(
                {
                    "rollno": rollno,
                    "name": name,
                    "parent_email": parent_email,
                    "present": present,
                    "total": n,
                    "pct": pct,
                }
            )
    return sorted(out, key=lambda r: r["pct"])


def lecture_breakdown(session, Attendance, course):
    """Distinct students present per lecture number."""
    rows = (
        session.query(
            Attendance.lecture_no,
            func.count(func.distinct(Attendance.rollno)),
        )
        .filter_by(course=course)
        .group_by(Attendance.lecture_no)
        .order_by(Attendance.lecture_no)
        .all()
    )
    return [{"lecture": lec, "present": p} for lec, p in rows]


def faculty_courses(session, Faculty, email):
    """Courses taught by a faculty member (for the course picker)."""
    fac = session.query(Faculty).filter_by(email=email).first()
    if not fac:
        return []
    if getattr(fac, "is_admin", False):
        rows = session.query(Attendance.course).distinct().all()
        return sorted(r[0] for r in rows if r[0])
    return [fac.course] if fac.course else []
