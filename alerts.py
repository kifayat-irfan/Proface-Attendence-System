"""Smart alerts engine (Phase 8).

Proactive notifications the system generates on its own:
- absence_streak: student absent 3+ consecutive school days
- low_attendance: student dropped below 75% overall
- unknown_face: unrecognized face seen repeatedly (placeholder hook)

check_smart_alerts() is idempotent — it never creates duplicate alerts for
the same (type, rollno, day).
"""
from datetime import date, timedelta


REENROLL_AFTER_DAYS = 365  # face templates older than this trigger a reminder


def check_smart_alerts(db, Student, Attendance, Alert, FaceTemplate=None):
    """Scan for alert conditions. Returns number of NEW alerts created.

    Designed to run periodically (cron, admin dashboard visit, or the
    /api/v1/alerts/check endpoint). Throttle upstream — this function itself
    is cheap but not free.
    """
    created = 0
    today = date.today()

    def _exists(alert_type, rollno):
        return Alert.query.filter_by(
            alert_type=alert_type, rollno=rollno,
            is_dismissed=False).filter(
            Alert.created_at >= today - timedelta(days=1)).first() is not None

    students = Student.query.all()
    if not students:
        return 0

    # --- recent school days (last 5 calendar days with any attendance) ---
    recent_days = sorted({
        a.marked_date for a in
        Attendance.query.filter(
            Attendance.marked_date >= today - timedelta(days=10)).all()
    }, reverse=True)[:5]

    for s in students:
        # 1. Absence streak: present on none of the last 3 school days
        if len(recent_days) >= 3:
            present = {a.marked_date for a in Attendance.query.filter_by(
                rollno=s.rollno).filter(
                Attendance.marked_date.in_(recent_days[:3])).all()}
            if not present and not _exists('absence_streak', s.rollno):
                db.session.add(Alert(
                    alert_type='absence_streak',
                    title=f'{s.name} absent {len(recent_days[:3])} days in a row',
                    details=(f'Roll no {s.rollno} has no attendance on '
                             f'{", ".join(d.isoformat() for d in recent_days[:3])}. '
                             f'Consider contacting the student/parent.'),
                    rollno=s.rollno))
                created += 1

        # 2. Low attendance: below 75% overall
        total = Attendance.query.filter_by(rollno=s.rollno).count()
        if total >= 5:  # need a minimum sample
            # distinct lecture-days as denominator proxy
            all_days = db.session.query(Attendance.marked_date).distinct().count()
            pct = (total / all_days * 100) if all_days else 100
            if pct < 75 and not _exists('low_attendance', s.rollno):
                db.session.add(Alert(
                    alert_type='low_attendance',
                    title=f'{s.name} attendance at {pct:.0f}% (below 75%)',
                    details=(f'Roll no {s.rollno}: {total}/{all_days} lectures. '
                             f'At risk of falling below the requirement.'),
                    rollno=s.rollno))
                created += 1

    # 3. Face aging: templates older than a year need re-enrollment
    if FaceTemplate is not None:
        cutoff = today - timedelta(days=REENROLL_AFTER_DAYS)
        # oldest template per student
        seen = set()
        for tpl in FaceTemplate.query.all():
            if tpl.rollno in seen:
                continue
            seen.add(tpl.rollno)
            if tpl.created_at and tpl.created_at.date() < cutoff \
                    and not _exists('reenroll', tpl.rollno):
                s = Student.query.filter_by(rollno=tpl.rollno).first()
                name = s.name if s else f'Roll no {tpl.rollno}'
                age_days = (today - tpl.created_at.date()).days
                db.session.add(Alert(
                    alert_type='reenroll',
                    title=f'{name}: face photo is {age_days} days old',
                    details=(f'Faces change over time. Re-enroll {name} '
                             f'(roll no {tpl.rollno}) with fresh photos for '
                             f'best accuracy.'),
                    rollno=tpl.rollno))
                created += 1

    if created:
        db.session.commit()
    return created
