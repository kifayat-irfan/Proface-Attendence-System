"""Tests for Phase 8b: face-aging reminders + tenant slug.

Light deps only.
"""
import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alerts import check_smart_alerts, REENROLL_AFTER_DAYS


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


class FakeDB:
    def __init__(self):
        self.added = []

    @property
    def session(self):
        return self

    def add(self, o):
        self.added.append(o)

    def commit(self):
        pass


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Col:
    def __ge__(self, o): return True
    def __le__(self, o): return True
    def __eq__(self, o): return True
    def in_(self, seq): return True


class _AlertModel:
    query = FakeQuery([])
    created_at = _Col(); alert_type = _Col()
    rollno = _Col(); is_dismissed = _Col()
    def __init__(self, **kw): self.__dict__.update(kw)


class TestFaceAging:
    def _run(self, template_age_days):
        db = FakeDB()
        s = Obj(rollno=5, name='Sara')
        Student = type('S', (), {'query': FakeQuery([s])})
        Attendance = type('A', (), {'query': FakeQuery([]),
                                    'marked_date': _Col()})
        tpl = Obj(rollno=5,
                  created_at=datetime.now() - timedelta(days=template_age_days))
        FaceTemplate = type('FT', (), {'query': FakeQuery([tpl])})
        n = check_smart_alerts(db, Student, Attendance, _AlertModel,
                               FaceTemplate)
        return n, db

    def test_old_template_triggers_reenroll(self):
        n, db = self._run(REENROLL_AFTER_DAYS + 30)
        assert n == 1
        assert db.added[0].alert_type == 'reenroll'
        assert '5' in db.added[0].details

    def test_fresh_template_no_alert(self):
        n, db = self._run(100)
        assert n == 0

    def test_threshold_is_one_year(self):
        assert REENROLL_AFTER_DAYS == 365


class TestTenantSlug:
    def test_get_tenant_default(self, monkeypatch):
        monkeypatch.delenv('TENANT_SLUG', raising=False)
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        assert "os.environ.get('TENANT_SLUG', 'default')" in src

    def test_tenant_in_backup_filename(self):
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        assert 'proface-backup-{get_tenant()}-' in src

    def test_tenant_in_api_health(self):
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        assert '"tenant": get_tenant()' in src

    def test_multi_compose_exists(self):
        p = os.path.join(os.path.dirname(__file__), '..',
                         'docker-compose.multi.yml')
        assert os.path.isfile(p)
        content = open(p).read()
        assert 'TENANT_SLUG: kkkuk' in content
        assert 'TENANT_SLUG: uop' in content
