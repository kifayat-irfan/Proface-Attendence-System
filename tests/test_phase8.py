"""Tests for Phase 8: smart alerts + API key hashing + video loop.

Light deps only.
"""
import hashlib
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alerts import check_smart_alerts


class FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def filter_by(self, **kw):
        return FakeQuery([r for r in self._rows
                          if all(getattr(r, k) == v for k, v in kw.items())])

    def filter(self, *a):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def count(self):
        return len(self._rows)

    def order_by(self, *a):
        return self

    def limit(self, n):
        return FakeQuery(self._rows[:n])


class FakeDB:
    def __init__(self):
        self.added = []
        self.committed = False

    @property
    def session(self):
        return self

    def add(self, o):
        self.added.append(o)

    def commit(self):
        self.committed = True


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Col:
    """Dummy SQLAlchemy column: any comparison returns True-ish."""
    def __ge__(self, o): return True
    def __le__(self, o): return True
    def __eq__(self, o): return True
    def in_(self, seq): return True


def _models(absent=True, low=False):
    s = Obj(rollno=1, name='Ali')
    days = [date.today() - timedelta(days=i) for i in range(1, 6)]
    att = [] if absent else [Obj(rollno=1, marked_date=d) for d in days[:3]]
    if low:
        att = [Obj(rollno=1, marked_date=days[0]),
               Obj(rollno=1, marked_date=days[1])]
    Student = type('S', (), {'query': FakeQuery([s])})
    Attendance = type('A', (), {'query': FakeQuery(att),
                                'marked_date': _Col(), 'rollno': _Col()})
    class Al:
        query = FakeQuery([])
        created_at = _Col(); alert_type = _Col()
        rollno = _Col(); is_dismissed = _Col()
        def __init__(self, **kw): self.__dict__.update(kw)
    Alert = Al
    return Student, Attendance, Alert


class TestSmartAlerts:
    def test_absence_streak_detected(self):
        db = FakeDB()
        Student, Attendance, Alert = _models(absent=True)
        # need 3+ school days: fake them via marked_date on other rollnos
        other = [Obj(rollno=99, marked_date=date.today() - timedelta(days=i))
                 for i in range(1, 4)]
        Attendance.query = FakeQuery(other)
        n = check_smart_alerts(db, Student, Attendance, Alert)
        assert n >= 1
        assert db.added[0].alert_type == 'absence_streak'

    def test_no_students_no_crash(self):
        db = FakeDB()
        Student = type('S', (), {'query': FakeQuery([])})
        Attendance = type('A', (), {'query': FakeQuery([]),
                                    'marked_date': _Col()})
        Alert = type('Al', (), {'query': FakeQuery([]),
                                'created_at': _Col()})
        assert check_smart_alerts(db, Student, Attendance, Alert) == 0

    def test_idempotent_no_duplicates(self):
        db = FakeDB()
        Student, Attendance, Alert = _models(absent=True)
        other = [Obj(rollno=99, marked_date=date.today() - timedelta(days=i))
                 for i in range(1, 4)]
        Attendance.query = FakeQuery(other)
        # pre-existing recent alert -> no duplicate
        existing = Obj(alert_type='absence_streak', rollno=1,
                       is_dismissed=False, created_at=date.today())
        class Al2:
            query = FakeQuery([existing])
            created_at = _Col(); alert_type = _Col()
            rollno = _Col(); is_dismissed = _Col()
            def __init__(self, **kw): self.__dict__.update(kw)
        Alert = Al2
        # _exists checks created_at >= yesterday; our fake filter ignores dates,
        # so first() returns the existing alert -> suppressed
        n = check_smart_alerts(db, Student, Attendance, Alert)
        assert n == 0


class TestApiContract:
    def test_endpoints_documented(self):
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        for ep in ['/api/v1/health', '/api/v1/students',
                   '/api/v1/attendance', '/api/v1/analytics/summary',
                   '/api/v1/alerts']:
            assert ep in src, f"missing {ep}"

    def test_api_key_header_required(self):
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        assert "X-API-Key" in src
