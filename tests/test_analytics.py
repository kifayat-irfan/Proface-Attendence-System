"""Unit tests for ProFace analytics (analytics.py).

Uses an in-memory SQLite DB with a schema mirroring app.py's models.
Run:  python -m pytest tests/ -v
"""
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Date
from sqlalchemy.orm import declarative_base, sessionmaker

from analytics import (
    course_kpis,
    attendance_trend,
    defaulter_list,
    lecture_breakdown,
)

Base = declarative_base()


class Student(Base):
    __tablename__ = "Student"
    id = Column(Integer, primary_key=True)
    rollno = Column(Integer, unique=True)
    name = Column(String(80))
    email = Column(String(100))
    parent_email = Column(String(120))
    approved = Column(Boolean, default=True)


class Attendance(Base):
    __tablename__ = "Attendance"
    att_id = Column(Integer, primary_key=True)
    rollno = Column(Integer)
    course = Column(String(80))
    lecture_no = Column(Integer)
    marked_date = Column(Date)


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed(session):
    # 4 approved students, course CS101, 4 lectures held
    for i, name in enumerate(["Ali", "Sara", "Omar", "Zara"], start=1):
        session.add(Student(rollno=100 + i, name=name,
                            email=f"s{i}@x.com",
                            parent_email=f"p{i}@x.com" if i != 4 else None,
                            approved=True))
    today = date.today()
    # Ali: all 4 lectures; Sara: 3; Omar: 1; Zara: 0
    marks = [(101, 1), (101, 2), (101, 3), (101, 4),
             (102, 1), (102, 2), (102, 3),
             (103, 2)]
    for rollno, lec in marks:
        session.add(Attendance(rollno=rollno, course="CS101",
                               lecture_no=lec, marked_date=today))
    session.commit()


def test_course_kpis():
    s = make_session()
    seed(s)
    k = course_kpis(s, Student, Attendance, "CS101")
    assert k["students"] == 4
    assert k["lectures"] == 4
    # 8 marks / (4 students * 4 lectures) = 50%
    assert k["avg_attendance_pct"] == 50.0


def test_defaulter_list():
    s = make_session()
    seed(s)
    d = defaulter_list(s, Student, Attendance, "CS101", threshold=75.0)
    rollnos = [r["rollno"] for r in d]
    assert 101 not in rollnos  # 100% - not a defaulter
    assert 102 not in rollnos  # 75% - at threshold, not below
    assert rollnos == [104, 103]  # worst first: 0%, then 25%
    assert d[0]["pct"] == 0.0
    assert d[1]["pct"] == 25.0
    assert d[0]["parent_email"] is None  # Zara has no parent email


def test_attendance_trend():
    s = make_session()
    seed(s)
    t = attendance_trend(s, Student, Attendance, "CS101", days=14)
    assert len(t) == 1
    assert t[0]["present"] == 3  # Ali, Sara, Omar distinct today
    assert t[0]["pct"] == 75.0


def test_lecture_breakdown():
    s = make_session()
    seed(s)
    b = lecture_breakdown(s, Attendance, "CS101")
    assert b == [
        {"lecture": 1, "present": 2},
        {"lecture": 2, "present": 3},
        {"lecture": 3, "present": 2},
        {"lecture": 4, "present": 1},
    ]


def test_empty_course():
    s = make_session()
    seed(s)
    k = course_kpis(s, Student, Attendance, "NOPE")
    assert k["lectures"] == 0
    assert k["avg_attendance_pct"] == 0.0
    assert defaulter_list(s, Student, Attendance, "NOPE") == []
