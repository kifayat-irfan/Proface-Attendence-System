import time
import os
from datetime import datetime, timedelta
from scipy.spatial import distance as dist
import requests
from keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, flash, session, Response, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import cv2
import dlib
import numpy as np
from geopy.distance import geodesic
import face_recognition

from security import hash_password, verify_password, is_hashed, is_live_face, LIVENESS_THRESHOLD
from recognition import DlibEngine, get_engine
from quality import assess_photo, issue_text
from stream import shared_stream
from alerts import check_smart_alerts
from analytics import course_kpis, attendance_trend, defaulter_list, lecture_breakdown
from notify import send_absence_alert, smtp_configured


# Paths to cascade files and emotion model
current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_face_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
cascade_eye_path = os.path.join(current_dir,  'haarcascade_eye.xml')
emotion_model_path = os.path.join(current_dir, 'emotion_model.h5')
# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cascade_face_path)
eye_cascade = cv2.CascadeClassifier(cascade_eye_path)

# Load emotion recognition model
emotion_model = load_model(emotion_model_path)

# Anti-spoofing CNN: 64x64 RGB face crop -> [P(fake), P(real)].
# This model shipped with the project but was never wired in - now it is.
# Trained with flow_from_directory (alphabetical classes), so index 1 = real.
# If live faces ever score as "fake", flip LIVENESS_REAL_INDEX to 0.
liveness_cnn = load_model(os.path.join(current_dir, 'liveness_model.h5'))
LIVENESS_INPUT_SIZE = (64, 64)
LIVENESS_REAL_INDEX = 1
# LIVENESS_THRESHOLD + is_live_face() live in security.py (unit-tested).



# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#################################################################################
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-me')
# --- Session & cookie hardening ---
app.config['SESSION_COOKIE_HTTPONLY'] = True   # JS can't read the session cookie (XSS mitigation)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF mitigation for cross-site POSTs
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# --- Rate limiting (brute-force protection on auth endpoints) ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per hour"],
    storage_uri="memory://",
)
# Database configuration: DATABASE_URL env for PostgreSQL, else local SQLite.
# Example: DATABASE_URL=postgresql://user:pass@localhost:5432/proface
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'sqlite:///' + os.path.join(basedir, 'instance', 'site.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}


db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import models here as to avoid circular import issue
class Student(db.Model):
    __tablename__ = 'Student'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    rollno = db.Column(db.Integer, unique=True, nullable=False)
    name = db.Column(db.String(80))
    semester = db.Column(db.String(80))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(255))  # hashed (scrypt); widened from 80
    parent_email = db.Column(db.String(120))  # for absence alerts to parents
    face_embedding = db.Column(db.LargeBinary)  # cached face embedding (float32 bytes)
    embedding_dim = db.Column(db.Integer)  # 128 (dlib) or 512 (arcface)
    embedding_engine = db.Column(db.String(20))  # engine that produced the embedding
    pic_path = db.Column(db.Text)
    registered_on = db.Column(db.DateTime)
    approved = db.Column(db.Boolean, default=False)  # New column


class Faculty(db.Model):
    __tablename__ = 'Faculty'
    __table_args__ = {'extend_existing': True}

    f_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    course = db.Column(db.String(80))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(255))  # hashed (scrypt); widened from 80
    is_admin = db.Column(db.Boolean, default=False)
    registered_on = db.Column(db.DateTime)
    approved = db.Column(db.Boolean, default=False)  # New column

class Attendance(db.Model):
    __tablename__ = 'Attendance'
    __table_args__ = (
        UniqueConstraint('rollno', 'lecture_no', 'marked_date', name='uix_1'),  # Unique constraint
        {'extend_existing': True}
    )

    att_id = db.Column(db.Integer, primary_key=True)
    rollno = db.Column(db.Integer)
    course = db.Column(db.String(80))
    lecture_no = db.Column(db.Integer)
    marked_by = db.Column(db.String(80))
    marked_date = db.Column(db.Date)
    marked_time = db.Column(db.Time)
    name = db.Column(db.String(80))  # New column
    location = db.Column(db.String(80))  # New column


class AuditLog(db.Model):
    __tablename__ = 'AuditLog'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    actor_email = db.Column(db.String(120))
    actor_role = db.Column(db.String(20))  # student / faculty / admin / system
    action = db.Column(db.String(80), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))


class FaceTemplate(db.Model):
    """Multiple face embeddings per student (Phase 6: multi-angle enrollment).

    Each row is one verified enrollment photo (front/left/right/up/glasses).
    The recognizer matches against ALL templates — a student is recognized
    if ANY of their angle templates matches.
    """
    __tablename__ = 'FaceTemplate'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    rollno = db.Column(db.Integer, nullable=False, index=True)
    angle_label = db.Column(db.String(20), default='front')  # front/left/right/up/glasses
    embedding = db.Column(db.LargeBinary, nullable=False)
    dim = db.Column(db.Integer, nullable=False)  # 128 (dlib) / 512 (arcface)
    engine = db.Column(db.String(20), nullable=False)
    quality_score = db.Column(db.Integer)  # 0-100 from quality.py
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ApiKey(db.Model):
    """API keys for the REST API (Phase 7). Only the SHA-256 hash is stored —
    the raw key is shown once at creation and never again."""
    __tablename__ = 'ApiKey'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    key_hash = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(80))  # e.g. "University ERP"
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used_at = db.Column(db.DateTime)


class Alert(db.Model):
    """Proactive smart alerts (Phase 8): absence streaks, low attendance,
    repeated unknown faces. Generated by check_smart_alerts()."""
    __tablename__ = 'Alert'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(30), nullable=False)  # absence_streak/low_attendance/unknown_face
    title = db.Column(db.String(200), nullable=False)
    details = db.Column(db.Text)
    rollno = db.Column(db.Integer, index=True)
    is_read = db.Column(db.Boolean, default=False)
    is_dismissed = db.Column(db.Boolean, default=False)



    with app.app_context():
        
        db.create_all()
def _hash_api_key(raw):
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()


def require_api_key(f):
    """Protect REST endpoints with X-API-Key header."""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        raw = request.headers.get('X-API-Key', '')
        if not raw:
            return {"error": "X-API-Key header required"}, 401
        ak = ApiKey.query.filter_by(
            key_hash=_hash_api_key(raw), is_active=True).first()
        if not ak:
            return {"error": "invalid API key"}, 401
        ak.last_used_at = datetime.utcnow()
        db.session.commit()
        return f(*args, **kwargs)
    return wrapper


def get_tenant():
    """Tenant slug for this deployment (multi-university).

    Each university runs its own deployment (Docker + own database) with a
    unique TENANT_SLUG. The slug tags backups, audit logs and API responses
    so data is never mixed up between institutions.
    """
    return os.environ.get('TENANT_SLUG', 'default').strip().lower() or 'default'


@app.context_processor
def inject_branding():
    """INSTITUTION_NAME + TENANT_SLUG branding on every page."""
    return {'INSTITUTION_NAME': os.environ.get(
                'INSTITUTION_NAME', 'ProFace Attendance System'),
            'TENANT_SLUG': get_tenant()}


def log_audit(action, details="", role=None):
    """Write an audit trail entry. Never raises - auditing must not break app flow."""
    try:
        email = session.get('uname') or session.get('roll_no') or 'anonymous'
        if role is None:
            role = 'admin' if session.get('is_admin') else (
                'faculty' if session.get('fty_logged_in') else (
                    'student' if session.get('std_logged_in') else 'anonymous'))
        db.session.add(AuditLog(
            actor_email=str(email),
            actor_role=role,
            action=action,
            details=f"[tenant:{get_tenant()}] {details}" if details else f"[tenant:{get_tenant()}]",
            ip_address=request.remote_addr,
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/login_student', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_student():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        student = Student.query.filter_by(email=email).first()

        if student is not None and verify_password(student.password, password):
            if not is_hashed(student.password):
                # Transparent upgrade of legacy plaintext password
                student.password = hash_password(password)
                db.session.commit()
            
            
            if not student.approved:  # Prevent unapproved students from signing in
                flash('Approval pending. Please wait for admin approval.', 'warning')
                return render_template('login_student.html')
            else:
                session['std_logged_in'] = True
                uname = email.split('@')[0]
                session['uname'] = uname
                session['roll_no'] = student.rollno

                flash('You are now logged in', 'success')
                log_audit('login', f'student {email} logged in')
                return redirect(url_for('student'))
        else:
            flash('Invalid email or password.', 'danger')

        return render_template('login_student.html')

    return render_template('login_student.html')


@app.route('/login_faculty', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_faculty():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        faculty = Faculty.query.filter_by(email=email).first()

        if faculty is not None and verify_password(faculty.password, password):
            if not is_hashed(faculty.password):
                # Transparent upgrade of legacy plaintext password
                faculty.password = hash_password(password)
                db.session.commit()
            # Admin login handling
            if faculty.is_admin:
                session['fty_logged_in'] = True
                session['uname'] = email.split('@')[0]
                session['name'] = faculty.name
                session['is_admin'] = True
                flash('Admin login successful', 'success')
                log_audit('login', f'admin {email} logged in', role='admin')
                return redirect(url_for('faculty'))  # Redirect to faculty dashboard
            
            # Non-admin faculty handling
            if not faculty.approved:
                flash('Approval pending. Please wait for admin approval.', 'warning')
                return render_template('login_faculty.html')

            # Regular faculty login handling
            session['fty_logged_in'] = True
            session['uname'] = email.split('@')[0]
            session['name'] = faculty.name
            session['course'] = faculty.course
            flash('You are now logged in', 'success')
            log_audit('login', f'faculty {email} logged in')
            return redirect(url_for('faculty'))  # Redirect to faculty dashboard

        else:
            flash('Invalid email or password.', 'danger')
            return render_template('login_faculty.html')

    return render_template('login_faculty.html')

@app.route('/signup_student', methods=['GET', 'POST'])
def signup_student():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        rollno = request.form['rollno']
        name = request.form['name']

        existing_user = Student.query.filter_by(email=email).first()

        if existing_user is None:
            new_student = Student(email=email, password=hash_password(password), rollno=rollno, name=name)
            db.session.add(new_student)
            db.session.commit()

            flash('Signup request submitted. Awaiting admin approval.', 'success')
            return redirect(url_for('login_student'))
        else:
            flash('Email already exists. Please log in.', 'danger')
            return redirect(url_for('login_student'))

    return render_template('signup_student.html')

@app.route('/signup_faculty', methods=['GET', 'POST'])
def signup_faculty():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        course = request.form['course']
        is_admin = 'is_admin' in request.form

        existing_user = Faculty.query.filter_by(email=email).first()

        if existing_user is None:
            new_faculty = Faculty(email=email, password=hash_password(password), name=name, course=course, is_admin=is_admin)
            db.session.add(new_faculty)
            db.session.commit()

            flash('Signup request submitted. Awaiting admin approval.', 'success')
            return redirect(url_for('login_faculty'))
        else:
            flash('Email already exists. Please log in.', 'danger')
            return redirect(url_for('login_faculty'))

    return render_template('signup_faculty.html')



def is_student_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'std_logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login!', 'danger')
            return redirect(url_for('login_student'))
    return wrap


def is_faculty_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'fty_logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login!', 'danger')
            return redirect(url_for('login_faculty'))
    return wrap
@app.route('/admin/approve_requests', methods=['GET', 'POST'])
@is_faculty_logged_in
def approve_requests():
    if 'is_admin' not in session:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login_faculty'))

    pending_students = Student.query.filter_by(approved=False).all()
    pending_faculty = Faculty.query.filter_by(approved=False).all()

    if request.method == 'POST':
        user_type = request.form['user_type']
        user_id = request.form['user_id']

        if user_type == 'student':
            student = Student.query.get(user_id)
            student.approved = True
        elif user_type == 'faculty':
            faculty = Faculty.query.get(user_id)
            faculty.approved = True

        db.session.commit()
        flash('User approved successfully.', 'success')
        return redirect(url_for('approve_requests'))

    return render_template('approve_requests.html', pending_students=pending_students, pending_faculty=pending_faculty)


@app.route('/student')
@is_student_logged_in
def student():
    courses = Attendance.query.filter_by(rollno=session['roll_no']).with_entities(
        Attendance.course).distinct().order_by(Attendance.course).all()
    lectures_count = []
    for course in courses:
        lec_count = Attendance.query.filter_by(
            rollno=session['roll_no'], course=course[0]).count()
        lectures_count.append(lec_count)
    app.logger.info(lectures_count)
    return render_template('dashboard_student.html', courses=courses, lectures_count=lectures_count)


@app.route('/my_attendance')
@is_student_logged_in
def view_attendance():
    roll_no = session.get('roll_no')  # Retrieve the student's roll number

    if not roll_no:
        return redirect(url_for('login_student'))  # Redirect if roll number is not set

    # Query to get distinct courses
    unique_courses = db.session.query(Attendance.course).filter_by(
        rollno=roll_no).distinct().all()

    # If no courses are found, handle accordingly
    if not unique_courses:
        return render_template('dashboard_student.html', courses=[], lectures_count={})

    # Query to get attendance counts for each course
    attendance_data = db.session.query(
        Attendance.course,
        db.func.count(Attendance.rollno).label('lectures_count')
    ).filter_by(rollno=roll_no) \
     .group_by(Attendance.course).all()

    # Convert attendance data to a dictionary
    lectures_count = {course: 0 for course, in unique_courses}
    for course, count in attendance_data:
        lectures_count[course] = count

    return render_template('dashboard_student.html', courses=[course for course, in unique_courses], lectures_count=lectures_count)

@app.route('/student_registration', methods=['GET', 'POST'])
@is_faculty_logged_in
def register_student():
    if request.method == 'POST':
        data = Student.query.filter_by(email=request.form['email']).first()
        if data is None:
            new_student = Student(
                rollno=request.form['rollno'],
                name=request.form['name'],
                semester=request.form['semester'],
                email=request.form['email'],
                password=hash_password(request.form['password']),
                parent_email=request.form.get('parent_email') or None,
                pic_path=f'static/images/users/{request.form["rollno"]}.jpg',
                registered_on=datetime.now()
            )
            db.session.add(new_student)
            db.session.commit()

            # Phase 6: multi-angle enrollment — photo saved as {rollno}__{angle}.jpg
            angle = session.pop('quality_angle', 'front')
            quality = session.pop('quality', {})
            photo_name = f'{request.form["rollno"]}__{angle}.jpg'
            if os.path.isfile('static/images/users/temp.jpg'):
                os.rename('static/images/users/temp.jpg',
                          f'static/images/users/{photo_name}')
            if 'img_captured' in session:
                session.pop('img_captured')
            # Make the new student recognizable immediately (no restart);
            # template saved in DB with its quality score
            new_photo = os.path.join(known_faces_dir, photo_name)
            if os.path.isfile(new_photo):
                add_known_face(new_photo, new_student, angle_label=angle,
                               quality_score=quality.get('score'))
            flash('Student registration successful', 'success')
            return render_template('register_student.html')
        else:
            flash('Student with this email already exists!', 'danger')

    if os.path.isfile('static/images/users/temp.jpg'):
        temp_pic = True
    else:
        temp_pic = False

    return render_template('register_student.html', temp_pic=temp_pic)


_last_alert_check = 0

def _maybe_check_alerts():
    """Run smart-alert scan at most once per hour (cheap, idempotent)."""
    global _last_alert_check
    if time.time() - _last_alert_check > 3600:
        _last_alert_check = time.time()
        try:
            n = check_smart_alerts(db, Student, Attendance, Alert, FaceTemplate)
            if n:
                print(f"Smart alerts: {n} new")
        except Exception as e:
            print(f"Alert check failed: {e}")


@app.route('/faculty')
@is_faculty_logged_in
def faculty():
    lectures_count = Attendance.query.filter_by(
    marked_by=session['name']).with_entities(Attendance.lecture_no).distinct().count()
    students_count = Student.query.with_entities(Student.rollno).count()
    faculty_count = Faculty.query.with_entities(Faculty.f_id).count()
    
    return render_template('dashboard_faculty.html', lectures_count=lectures_count, students_count=students_count, faculty_count=faculty_count)


@app.route('/faculty_registration', methods=['GET', 'POST'])
@is_faculty_logged_in
def register_faculty():
    if request.method == 'POST':
        data = Faculty.query.filter_by(email=request.form['email']).first()
        if 'isAdmin' in request.form:
            is_admin = True
        else:
            is_admin = False
        # if email does not already exist
        if data is None:
            faculty = Faculty(
                name=request.form['name'],
                course=request.form['course'],
                email=request.form['email'],
                password=hash_password(request.form['password']),
                is_admin=is_admin,
                registered_on=datetime.now()
            )
            db.session.add(faculty)
            db.session.commit()

            flash('Faculty registration successful', 'success')
            return render_template('register_faculty.html')
        else:
            flash('Faculty with this email already exists!', 'danger')
    return render_template('register_faculty.html')
 

@app.route('/faculty_list')
@is_faculty_logged_in
def faculty_list():
    faculty = Faculty.query.all()
    return render_template('faculty_info.html', faculty=faculty)

@app.route('/students_list')
@is_faculty_logged_in
def students_list():
    students = Student.query.all()
    return render_template('student_info.html', students=students)

@app.route('/lectures_list')
@is_faculty_logged_in
def lectures_list():
    # Fetch the list of lectures with associated details
    lectures = db.session.query(
        Attendance.course,
        
        Attendance.marked_date,
        Attendance.marked_time
    ).distinct().all()
    
    # Convert query results to a list of dictionaries for easier template processing
    lectures_list = [
        {
            'course': lecture.course,
            
            'date': lecture.marked_date,
            'time': lecture.marked_time
        } for lecture in lectures
    ]
    
    return render_template('lectures_list.html', lectures=lectures_list)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
@is_faculty_logged_in
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    rollno = student.rollno
    db.session.delete(student)
    db.session.commit()
    log_audit('student_deleted', f'rollno {rollno}')
    flash('Student deleted successfully', 'success')
    return redirect(url_for('students_list'))
@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@is_faculty_logged_in
def edit_student(student_id):
    student = Student.query.get_or_404(student_id)
    if request.method == 'POST':
        student.name = request.form['name']
        student.rollno = request.form['rollno']
        student.semester = request.form['semester']
        student.email = request.form['email']
        new_pw = request.form.get('password', '').strip()
        if new_pw:
            student.password = hash_password(new_pw)
        student.parent_email = request.form.get('parent_email') or None
        db.session.commit()
        log_audit('student_edited', f'rollno {student.rollno}')
        flash('Student details updated successfully', 'success')
        return redirect(url_for('students_list'))
    return render_template('edit_student.html', student=student)
@app.route('/edit_attendance/<int:att_id>', methods=['GET', 'POST'])
@is_faculty_logged_in
def edit_attendance(att_id):
    attendance = Attendance.query.get_or_404(att_id)
    
    if request.method == 'POST':
        # Update attendance record based on form input
        attendance.rollno = request.form['rollno']
        attendance.course = request.form['course']
        attendance.lecture_no = request.form['lecture_no']
        attendance.marked_by = request.form['marked_by']
        attendance.marked_date = datetime.strptime(request.form['marked_date'], '%Y-%m-%d').date()
        attendance.marked_time = datetime.strptime(request.form['marked_time'], '%H:%M:%S').time()
        attendance.name = request.form['name']
        attendance.location = request.form['location']
        
        db.session.commit()
        log_audit('attendance_edited', f'att_id {att_id}')
        flash('Attendance record updated successfully', 'success')
        return redirect(url_for('view_lectures_attendance'))

    return render_template('edit_attendance.html', attendance=attendance)

@app.route('/delete_attendance/<int:att_id>', methods=['POST'])
@is_faculty_logged_in
def delete_attendance(att_id):
    attendance = Attendance.query.get_or_404(att_id)
    att_id = attendance.att_id
    db.session.delete(attendance)
    db.session.commit()
    log_audit('attendance_deleted', f'att_id {att_id}')
    flash('Attendance record deleted successfully', 'success')
    return redirect(url_for('view_lectures_attendance'))


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route("/capture_image")
@is_faculty_logged_in
def capture_image():
    session['dt'] = datetime.now()
    # Phase 6: which angle is being captured (front/left/right/up/glasses)
    angle = request.args.get('angle', 'front')
    if angle not in ('front', 'left', 'right', 'up', 'glasses'):
        angle = 'front'
    session['capture_angle'] = angle
    path = 'static/images/users'
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Press c to capture image', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(os.path.join(path, 'temp.jpg'), frame)
            time.sleep(2)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Phase 6: quality gate — reject bad enrollment photos immediately
    temp_path = os.path.join(path, 'temp.jpg')
    img = cv2.imread(temp_path)
    if img is not None:
        result = assess_photo(img, recognition_engine)
        session['quality'] = {'score': result['score'], 'issues': result['issues']}
        if result['passed']:
            session['img_captured'] = True
            session['quality_angle'] = session.get('capture_angle', 'front')
            flash(f"Photo accepted (quality {result['score']}/100) ✅", 'success')
        else:
            session.pop('img_captured', None)
            flash(f"Photo rejected: {issue_text(result['issues'])}", 'danger')
    else:
        session.pop('img_captured', None)
        flash('Capture failed — try again.', 'danger')

    return redirect(url_for('register_student'))


@app.route('/attendance', methods=['GET', 'POST'])
@is_faculty_logged_in
def mark_attendance_1():
    std_reg = False
    std = Student.query.all()
    if len(std) > 0:
        std_reg = True
    if request.method == 'POST':
        session['lecture_no'] = request.form['lecture']
        return redirect(url_for('mark_attendance_2'))

    return render_template('mark_attendance_1.html', std_reg=std_reg)


@app.route('/attendance/fr')
@is_faculty_logged_in
def mark_attendance_2():
    return render_template('mark_attendance_2.html')


@app.route('/view_lectures_attendance/')
@is_faculty_logged_in
def view_lectures_attendance():
    lect_no = request.args.get('lecture')
    if lect_no is None:
        lect_no = 1
    # get unique lecture numbers marked by current faculty
    unique_lect_no = Attendance.query.filter_by(marked_by=session['name']).with_entities(
        Attendance.lecture_no).distinct().order_by(Attendance.lecture_no).all()
    # get the attendance for the specified lecture
    attendance = Attendance.query.filter_by(
        lecture_no=lect_no, marked_by=session['name']).all()

    return render_template('view_lecture_attendance.html', unique_lect_no=unique_lect_no, attendance=attendance)

# TODO download attendance for each lecture as marked by the teacher


@app.route('/facultydownloadcsv', defaults={'lect_no': 1})
@app.route('/facultydownloadcsv/<int:lect_no>')
@is_faculty_logged_in
def download_attendance_csv(lect_no):
    headings = 'Roll_no,Course,Lecture_no,Marked_by,Marking_date,Marking_Time\n'
    attendance = Attendance.query.filter_by(
        lecture_no=lect_no, marked_by=session['name']).all()
    rows = ''
    for a in attendance:
        rows += str(a.rollno)+','+str(a.course)+','+str(a.lecture_no)+',' + \
            (a.marked_by)+','+str(a.marked_date)+','+str(a.marked_time)+' \n'
    csv = headings+rows
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=attendance.csv"})


@app.route('/studentdownloadcsv', defaults={'course': None})
@app.route('/studentdownloadcsv/<string:course>')
@is_student_logged_in
def download_student_attendance_csv(course):
    headings = 'Roll_no,Course,Lecture_no,Marked_by,Marking_date,Marking_Time\n'
    attendance = Attendance.query.filter_by(
        rollno=session['roll_no'], course=course).all()
    rows = ''
    for a in attendance:
        rows += str(a.rollno)+','+str(a.course)+','+str(a.lecture_no)+',' + \
            (a.marked_by)+','+str(a.marked_date)+','+str(a.marked_time)+' \n'
    csv = headings+rows
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=my_attendance.csv"})




# ================= Analytics (Phase 3: Intelligence) =================
@app.route('/analytics')
@is_faculty_logged_in
def analytics_dashboard():
    courses = _my_courses()
    selected = request.args.get('course') or (courses[0] if courses else None)
    return render_template('analytics.html', courses=courses, selected=selected,
                           smtp_ready=smtp_configured())


def _my_courses():
    """Courses visible to the logged-in faculty (all for admins)."""
    if session.get('is_admin'):
        rows = db.session.query(Attendance.course).distinct().all()
        return sorted(r[0] for r in rows if r[0])
    name = session.get('name')
    fac = Faculty.query.filter_by(name=name).first()
    return [fac.course] if fac and fac.course else []


@app.route('/api/analytics')
@is_faculty_logged_in
def api_analytics():
    course = request.args.get('course')
    if not course:
        return {"error": "course required"}, 400
    return {
        "kpis": course_kpis(db.session, Student, Attendance, course),
        "trend": attendance_trend(db.session, Student, Attendance, course),
        "defaulters": defaulter_list(db.session, Student, Attendance, course),
        "lectures": lecture_breakdown(db.session, Attendance, course),
    }


@app.route('/alert_absentees', methods=['POST'])
@is_faculty_logged_in
def alert_absentees():
    course = request.form.get('course')
    if not course:
        flash('No course selected.', 'danger')
        return redirect(url_for('analytics_dashboard'))
    sent, skipped = 0, 0
    for d in defaulter_list(db.session, Student, Attendance, course):
        ok, _ = send_absence_alert(d["parent_email"], d["name"], course,
                                   d["pct"], d["present"], d["total"])
        if ok:
            sent += 1
        else:
            skipped += 1
    if sent:
        flash(f'Absence alerts sent to {sent} parent(s).', 'success')
    if skipped:
        flash(f'{skipped} skipped (no parent email or SMTP not configured).', 'warning')
    if not sent and not skipped:
        flash('No defaulters found - nothing to send.', 'info')
    return redirect(url_for('analytics_dashboard', course=course))


@app.route('/audit_log')
@is_faculty_logged_in
def audit_log():
    if not session.get('is_admin'):
        flash('Admins only.', 'danger')
        return redirect(url_for('faculty'))
    page = request.args.get('page', 1, type=int)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).paginate(
        page=page, per_page=50, error_out=False)
    return render_template('audit_log.html', logs=logs)


@app.route('/liveness_check')
@is_faculty_logged_in
def liveness_check():
    """Real-world verification guide for the anti-spoofing CNN."""
    return render_template('liveness_check.html')


@app.route('/api_keys', methods=['GET', 'POST'])
@is_faculty_logged_in
def api_keys():
    """Admin: generate / revoke REST API keys. Raw key shown ONCE."""
    if not session.get('is_admin'):
        flash('Admins only.', 'danger')
        return redirect(url_for('faculty'))
    new_raw = None
    if request.method == 'POST':
        import secrets
        name = request.form.get('name', 'Unnamed integration').strip() or 'Unnamed'
        new_raw = secrets.token_urlsafe(32)
        db.session.add(ApiKey(key_hash=_hash_api_key(new_raw), name=name))
        db.session.commit()
        log_audit('api_key_created', name, role='admin')
        flash('API key created — copy it now, it will never be shown again!',
              'warning')
    if request.args.get('revoke'):
        ak = ApiKey.query.get(int(request.args['revoke']))
        if ak:
            ak.is_active = False
            db.session.commit()
            log_audit('api_key_revoked', ak.name, role='admin')
            flash('API key revoked.', 'success')
        return redirect(url_for('api_keys'))
    keys = ApiKey.query.order_by(ApiKey.created_at.desc()).all()
    return render_template('api_keys.html', keys=keys, new_raw=new_raw)


# ================= REST API v1 (Phase 7) =================
# Auth: X-API-Key header. Manage keys at /api_keys (admin).
# Example: curl -H "X-API-Key: <key>" /api/v1/students

@app.route('/api/v1/health')
def api_health():
    return {"status": "ok", "version": "2.0",
            "tenant": get_tenant(),
            "institution": os.environ.get('INSTITUTION_NAME', 'ProFace'),
            "engine": recognition_engine.name,
            "stream_running": shared_stream.stats["running"]}


@app.route('/api/v1/students')
@require_api_key
def api_students():
    students = Student.query.all()
    return {"students": [
        {"rollno": s.rollno, "name": s.name, "email": s.email,
         "semester": s.semester,
         "templates": FaceTemplate.query.filter_by(rollno=s.rollno).count()}
        for s in students]}


@app.route('/api/v1/students/<int:rollno>')
@require_api_key
def api_student_one(rollno):
    s = Student.query.filter_by(rollno=rollno).first()
    if not s:
        return {"error": "not found"}, 404
    return {"rollno": s.rollno, "name": s.name, "email": s.email,
            "semester": s.semester,
            "templates": [{"angle": t.angle_label,
                           "quality": t.quality_score,
                           "engine": t.engine}
                          for t in FaceTemplate.query.filter_by(rollno=rollno)]}


@app.route('/api/v1/attendance')
@require_api_key
def api_attendance_list():
    q = Attendance.query
    if request.args.get('date'):
        try:
            d = datetime.strptime(request.args['date'], '%Y-%m-%d').date()
            q = q.filter_by(marked_date=d)
        except ValueError:
            return {"error": "date must be YYYY-MM-DD"}, 400
    for f in ('course', 'lecture_no', 'rollno'):
        if request.args.get(f):
            q = q.filter_by(**{f: request.args[f]})
    rows = q.order_by(Attendance.marked_date.desc()).limit(500).all()
    return {"attendance": [
        {"id": r.att_id, "rollno": r.rollno, "name": r.name,
         "course": r.course, "lecture_no": r.lecture_no,
         "date": r.marked_date.isoformat() if r.marked_date else None,
         "time": r.marked_time.isoformat() if r.marked_time else None,
         "marked_by": r.marked_by, "location": r.location}
        for r in rows]}


@app.route('/api/v1/attendance', methods=['POST'])
@require_api_key
def api_attendance_mark():
    data = request.get_json(force=True, silent=True) or {}
    for f in ('rollno', 'course', 'lecture_no'):
        if f not in data:
            return {"error": f"missing field: {f}"}, 400
    try:
        d = datetime.strptime(
            data.get('date', datetime.now().date().isoformat()),
            '%Y-%m-%d').date()
    except ValueError:
        return {"error": "date must be YYYY-MM-DD"}, 400
    exists = Attendance.query.filter_by(
        rollno=data['rollno'], lecture_no=data['lecture_no'],
        marked_date=d).first()
    if exists:
        return {"marked": False, "reason": "already marked"}, 200
    s = Student.query.filter_by(rollno=data['rollno']).first()
    db.session.add(Attendance(
        name=s.name if s else 'Unknown', rollno=data['rollno'],
        course=data['course'], lecture_no=data['lecture_no'],
        marked_by='api', marked_date=d,
        marked_time=datetime.now().time(),
        location=data.get('location', 'via API')))
    db.session.commit()
    log_audit('attendance_marked_api',
              f"{data['rollno']} via REST API", role='system')
    return {"marked": True}, 201


@app.route('/api/v1/analytics/summary')
@require_api_key
def api_analytics_summary():
    from analytics import course_kpis
    return course_kpis()


@app.route('/alerts')
@is_faculty_logged_in
def alerts_page():
    alerts = Alert.query.filter_by(is_dismissed=False).order_by(
        Alert.created_at.desc()).limit(100).all()
    unread = sum(1 for a in alerts if not a.is_read)
    for a in alerts:
        a.is_read = True
    db.session.commit()
    return render_template('alerts.html', alerts=alerts, unread=unread)


@app.route('/alerts/dismiss/<int:alert_id>')
@is_faculty_logged_in
def dismiss_alert(alert_id):
    a = Alert.query.get(alert_id)
    if a:
        a.is_dismissed = True
        db.session.commit()
    return redirect(url_for('alerts_page'))


@app.route('/api/v1/alerts/check', methods=['POST'])
@require_api_key
def api_alerts_check():
    n = check_smart_alerts(db, Student, Attendance, Alert, FaceTemplate)
    log_audit('alerts_checked', f'{n} new alerts', role='system')
    return {"new_alerts": n}


@app.route('/api/v1/alerts')
@require_api_key
def api_alerts_list():
    alerts = Alert.query.filter_by(is_dismissed=False).order_by(
        Alert.created_at.desc()).limit(100).all()
    return {"alerts": [
        {"id": a.id, "type": a.alert_type, "title": a.title,
         "details": a.details, "rollno": a.rollno,
         "created_at": a.created_at.isoformat() if a.created_at else None,
         "is_read": a.is_read}
        for a in alerts]}


@app.route('/admin/backup')
@is_faculty_logged_in
def admin_backup():
    """Download a full JSON backup of all data (admin only).

    On Supabase/Postgres, automatic daily backups also run — this is the
    manual export for migrations and peace of mind.
    """
    if not session.get('is_admin'):
        flash('Admins only.', 'danger')
        return redirect(url_for('faculty'))
    import io, json
    data = {"exported_at": datetime.utcnow().isoformat(),
            "tenant": get_tenant(),
            "institution": os.environ.get('INSTITUTION_NAME', 'ProFace'),
            "tables": {}}
    for model, name in [(Student, 'students'), (Attendance, 'attendance'),
                        (Faculty, 'faculty'), (AuditLog, 'audit_log'),
                        (Alert, 'alerts')]:
        rows = []
        for r in model.query.all():
            d = {c.name: getattr(r, c.name) for c in r.__table__.columns
                 if c.name not in ('password', 'face_embedding', 'embedding',
                                   'key_hash')}
            # serialize dates
            for k, v in d.items():
                if hasattr(v, 'isoformat'):
                    d[k] = v.isoformat()
            rows.append(d)
        data["tables"][name] = rows
    buf = io.BytesIO(json.dumps(data, indent=2).encode())
    log_audit('backup_exported', f"{sum(len(v) for v in data['tables'].values())} rows",
              role='admin')
    return send_file(buf, mimetype='application/json', as_attachment=True,
                     download_name=f"proface-backup-{get_tenant()}-{date.today().isoformat()}.json")


@app.route("/logout")
def logout():
    if 'std_logged_in' or 'fty_logged_in' in session:
        session.clear()
    return redirect(url_for('index'))

# Emotion detection function
# --- Vision pipeline: one shared detector, one detection pass per frame ---
def detect_emotion(face_crop_bgr):
    """Emotion label for one BGR face crop (never the whole frame)."""
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    arr = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1) / 255.0
    return emotion_labels[int(np.argmax(emotion_model.predict(arr, verbose=0)))]


def cnn_liveness_score(face_crop_bgr):
    """P(real person) from the anti-spoofing CNN for one BGR face crop."""
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, LIVENESS_INPUT_SIZE)
    arr = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
    probs = liveness_cnn.predict(arr, verbose=0)[0]
    return float(probs[LIVENESS_REAL_INDEX])


def recognize_faces(frame):
    """Single detection pass per frame via the active recognition engine.
    Returns [{rollno|None, rect, encoding}] shared by recognition AND
    annotation, so the overlay can never mismatch the identities."""
    results = []
    for (x1, y1, x2, y2), encoding in recognition_engine.detect_and_encode(frame):
        results.append({
            "rollno": recognition_engine.match(encoding, face_registry),
            "rect": (x1, y1, x2, y2),
            "encoding": encoding,
        })
    return results

KARAK_LOCATION = (32.193, 70.0633)

def is_near_karak(location, threshold_km=10):
    from geopy.distance import great_circle
    distance = great_circle(location, KARAK_LOCATION).kilometers
    return distance <= threshold_km

def get_user_location():
    try:
        # Make a request to ipinfo.io to get location data based on IP address
        response = requests.get('https://ipinfo.io/json')
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        
        # Debugging: print the response data
        print("API Response Data:", data)

        # Extract location information
        location = data.get('loc', '0,0').split(',')
        country = data.get('country', 'Unknown Country')

        # Check if location has two values
        if len(location) == 2:
            user_location = (float(location[0]), float(location[1]))
            return {
                "location": user_location,
                "country": country,
                "is_within_university": is_near_karak(user_location)
            }
        else:
            return {"error": "Location data is not in expected format."}
    except requests.RequestException as e:
        # Handle any errors that occur during the request
        return {"error": str(e)}



# A simple counter and timer for blink detection
BLINK_THRESHOLD = 0.2
CONSEC_FRAMES = 3  # Number of consecutive frames to check for a blink
# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def get_eye_points(shape, start, end):
    points = []
    for i in range(start, end):
        point = (shape.part(i).x, shape.part(i).y)
        points.append(point)
    return np.array(points)
class BlinkDetector:
    def __init__(self):
        self.blink_start_time = None
        self.blinking = False
        self.frames_without_blink = 0

    def detect_blink(self, left_eye_pts, right_eye_pts):
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < BLINK_THRESHOLD:
            if not self.blinking:
                self.blink_start_time = time.time()
                self.blinking = True
        else:
            if self.blinking:
                blink_duration = time.time() - self.blink_start_time
                self.blinking = False
                if blink_duration >= 0.1:  # Consider a blink if duration is sufficient
                    self.frames_without_blink = 0  # Reset counter on blink
                    return True  # Detected a blink
        self.frames_without_blink += 1
        return False  # No blink detected
    
    def is_live(self):
        # If there has been a blink in the last few frames, assume live
        return self.frames_without_blink < CONSEC_FRAMES

blink_detector = BlinkDetector()

def detect_liveness(shape):
    left_eye_pts = get_eye_points(shape, 36, 42)
    right_eye_pts = get_eye_points(shape, 42, 48)

    if blink_detector.detect_blink(left_eye_pts, right_eye_pts):
        return True  # Blink detected, assume live
    
    return blink_detector.is_live()  # Check if there has been a blink recently

# Update the detect_faces_and_annotate function to include liveness detection
def detect_faces_and_annotate(frame, detections, location):
    """Annotate each detection with roll no, emotion and liveness.

    Works from the single detection pass produced by recognize_faces(), so the
    overlay always matches the recognized identities. Per-face results
    (emotion, CNN score, live flag) are written back into the detection dicts
    for the marking loop to use.

    Liveness is combined: the anti-spoofing CNN scores the face crop every
    frame, and the EAR blink detector confirms a recent blink. A face counts
    as live if EITHER fires — the CNN catches printed photos, the blink check
    catches replayed videos.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for det in detections:
        x1, y1, x2, y2 = det["rect"]
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        rollno = det["rollno"] if det["rollno"] else "Unknown"

        if crop.size:
            emotion = detect_emotion(crop)
            cnn_score = cnn_liveness_score(crop)
        else:
            emotion, cnn_score = "Unknown", 0.0
        shape = shape_predictor(rgb_frame, dlib.rectangle(x1, y1, x2, y2))
        blink_live = detect_liveness(shape)
        live = is_live_face(blink_live, cnn_score)

        det["emotion"] = emotion
        det["cnn_score"] = cnn_score
        det["live"] = live

        student_name = "Unknown Student"
        if rollno != "Unknown":
            student = Student.query.filter_by(rollno=rollno).first()
            if student:
                student_name = student.name

        box_color = (0, 255, 0) if live else (0, 0, 255)
        liveness_status = "Live" if live else "Fake - not marked"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f'Name: {student_name}', (x1, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
        cv2.putText(frame, f'Roll No: {rollno}', (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
        cv2.putText(frame, f'Liveness: {liveness_status} ({cnn_score:.2f})', (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    cv2.putText(frame, f'Location: {location}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame
# --- Face recognition engine (dlib default; RECOGNITION_ENGINE=arcface for ArcFace) ---
# dlib models loaded ONCE; absolute paths.
shape_predictor = dlib.shape_predictor(os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat"))
face_recognition_model = dlib.face_recognition_model_v1(
    os.path.join(current_dir, "dlib_face_recognition_resnet_model_v1.dat"))

_engine_choice = os.environ.get("RECOGNITION_ENGINE", "dlib").lower()
if _engine_choice == "arcface":
    try:
        recognition_engine = get_engine()  # builds ArcFaceEngine (downloads models once)
        print("Recognition engine: ArcFace (512-d)")
    except Exception as e:
        print(f"ArcFace unavailable ({e}); falling back to dlib")
        recognition_engine = DlibEngine(shape_predictor, face_recognition_model,
                                        dlib.get_frontal_face_detector())
else:
    recognition_engine = DlibEngine(shape_predictor, face_recognition_model,
                                    dlib.get_frontal_face_detector())
    print("Recognition engine: dlib (128-d)")

# --- Face template registry with DB embedding cache ---
# face_registry: [(rollno, np.ndarray embedding)] for the ACTIVE engine.
# Embeddings are cached in Student.face_embedding so restarts don't re-encode
# hundreds of photos (the old code re-encoded every photo on every startup).
face_registry = []
known_faces_dir = os.path.join(current_dir, "static", "images", "users")


def _emb_to_bytes(emb):
    return np.asarray(emb, dtype=np.float32).tobytes()


def _emb_from_bytes(data, dim):
    return np.frombuffer(data, dtype=np.float32).reshape(dim)


def _encode_photo(image_path):
    """Encode one photo with the active engine. Returns embedding or None."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hits = recognition_engine.detect_and_encode(img)
    return hits[0][1] if hits else None


def save_template(rollno, embedding, angle_label='front', quality_score=None):
    """Persist one face template (multi-angle enrollment, Phase 6)."""
    try:
        FaceTemplate.query.filter_by(
            rollno=rollno, angle_label=angle_label,
            engine=recognition_engine.name).delete()
        db.session.add(FaceTemplate(
            rollno=rollno, angle_label=angle_label,
            embedding=_emb_to_bytes(embedding),
            dim=recognition_engine.dim, engine=recognition_engine.name,
            quality_score=quality_score))
        db.session.commit()
        return True
    except Exception:
        db.session.rollback()
        return False


def add_known_face(image_path, student=None, angle_label='front', quality_score=None):
    """Add one photo to the registry (and save its template in the DB).

    Supports multi-angle filenames: {rollno}__{angle}.jpg
    Called at startup for every photo and right after a new student
    registers — no server restart needed.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    if '__' in basename:
        rollno, angle_label = basename.rsplit('__', 1)
    else:
        rollno = basename
    try:
        rollno_int = int(rollno)
    except ValueError:
        return False

    emb = None
    # 1. FaceTemplate rows first (instant, multi-angle)
    tpl = FaceTemplate.query.filter_by(
        rollno=rollno_int, angle_label=angle_label,
        engine=recognition_engine.name,
        dim=recognition_engine.dim).first() if student is not None else None
    if tpl:
        emb = _emb_from_bytes(tpl.embedding, tpl.dim)
    # 2. Legacy single-embedding cache on Student (migrate it)
    if emb is None and student is not None and student.face_embedding \
            and student.embedding_engine == recognition_engine.name \
            and student.embedding_dim == recognition_engine.dim:
        emb = _emb_from_bytes(student.face_embedding, student.embedding_dim)
        save_template(rollno_int, emb, angle_label, quality_score)
    # 3. Compute from photo and save
    if emb is None:
        emb = _encode_photo(image_path)
        if emb is not None:
            save_template(rollno_int, emb, angle_label, quality_score)
    if emb is not None:
        # Refresh ALL of this student's templates from the DB (keeps every
        # angle; replaces the re-saved one). Matcher tries all of them.
        face_registry[:] = [(r, e) for r, e in face_registry
                            if str(r) != str(rollno_int)]
        try:
            for tpl in FaceTemplate.query.filter_by(
                    rollno=rollno_int, engine=recognition_engine.name,
                    dim=recognition_engine.dim).all():
                face_registry.append(
                    (rollno_int, _emb_from_bytes(tpl.embedding, tpl.dim)))
        except Exception:
            face_registry.append((rollno_int, emb))
        return True
    return False


def load_known_faces():
    """(Re)load every registered student into the face registry.

    Multi-angle: each student may have several templates (front/left/right
    ...); the matcher tries all of them.
    """
    face_registry.clear()
    # 1. Fast path: all templates for the active engine from the DB
    try:
        rows = FaceTemplate.query.filter_by(
            engine=recognition_engine.name,
            dim=recognition_engine.dim).all()
        for tpl in rows:
            face_registry.append(
                (tpl.rollno, _emb_from_bytes(tpl.embedding, tpl.dim)))
        if face_registry:
            print(f"Face registry: {len(face_registry)} templates "
                  f"({recognition_engine.name}, from DB)")
            return
    except Exception as e:
        print(f"FaceTemplate load failed ({e}); falling back to photos")
    # 2. Slow path: encode from photos (also seeds the DB)
    if not os.path.isdir(known_faces_dir):
        return
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")) \
                and filename != "temp.jpg":
            basename = os.path.splitext(filename)[0]
            rollno = basename.rsplit('__', 1)[0] if '__' in basename else basename
            try:
                student = Student.query.filter_by(rollno=int(rollno)).first()
            except ValueError:
                student = None
            add_known_face(os.path.join(known_faces_dir, filename), student)


with app.app_context():
    load_known_faces()
print(f"Face registry: {len(face_registry)} templates ({recognition_engine.name})")

@app.route('/fr_attendance')
@is_faculty_logged_in
def mark_face_attendance():
    """Live face-attendance page (browser MJPEG stream). The old blocking
    cv2.imshow loop is gone — video now streams to the browser."""
    return redirect(url_for('live_marking'))


# ================= Live streaming (Phase 5) =================
# Camera source: CAMERA_SOURCE env — "0" for webcam, or an RTSP/HTTP URL
# for IP cameras. Examples: "0", "rtsp://user:pass@192.168.1.10/stream"
def get_camera_source():
    src = os.environ.get("CAMERA_SOURCE", "0").strip()
    return int(src) if src.isdigit() else src


def _mark_one_attendance(rollno, location_str, ctx=None):
    """Mark attendance for one roll number. Returns True if newly marked.

    ctx: explicit marking context (the shared-stream background thread has
    no Flask session, so it passes ctx explicitly).
    """
    student = Student.query.filter_by(rollno=rollno).first()
    student_name = student.name if student else 'Unknown Student'
    ctx = ctx or {}
    lecture_number = ctx.get('lecture_no') or session.get('lecture_no', 'Unknown Lecture')
    course = ctx.get('course') or session.get('course', 'Unknown')
    marked_by = ctx.get('marked_by') or session.get('name', 'Unknown')
    exists = Attendance.query.filter_by(
        rollno=rollno, lecture_no=lecture_number,
        marked_date=datetime.now().date()).first()
    if exists:
        return False
    db.session.add(Attendance(
        name=student_name, rollno=rollno, course=course,
        lecture_no=lecture_number, marked_by=marked_by,
        marked_date=datetime.now().date(), marked_time=datetime.now().time(),
        location=location_str))
    db.session.commit()
    try:
        log_audit("attendance_marked",
                  f"{rollno} marked for lecture {lecture_number} ({course})")
    except Exception:
        pass
    return True


def _shared_mark(rollno, location_str, ctx):
    """Attendance marking inside the stream thread (needs app context)."""
    with app.app_context():
        return _mark_one_attendance(rollno, location_str, ctx)


# Bind app functions into the shared stream (once, at startup)
shared_stream.bind(
    get_camera_source=get_camera_source,
    get_user_location=get_user_location,
    recognize_faces=recognize_faces,
    detect_faces_and_annotate=detect_faces_and_annotate,
    mark_attendance=_shared_mark,
)


def shared_frame_generator():
    """Per-viewer MJPEG generator — just relays the shared stream's frames."""
    try:
        while True:
            jpeg = shared_stream.get_jpeg()
            if jpeg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpeg + b'\r\n')
            time.sleep(0.04)  # ~25 fps cap per viewer; detection runs independently
    finally:
        shared_stream.stop()  # viewer disconnected


@app.route('/video_feed')
@is_faculty_logged_in
def video_feed():
    # Phase 7: shared stream — one camera/detection loop for ALL viewers.
    # Capture this faculty's marking context (the background thread has no session).
    ctx = {
        'lecture_no': session.get('lecture_no'),
        'course': session.get('course'),
        'marked_by': session.get('name'),
        'gps': session.get('gps'),
    }
    shared_stream.start(ctx)
    return Response(shared_frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live')
@is_faculty_logged_in
def live_marking():
    return render_template('live_marking.html',
                           camera_source=str(get_camera_source()))


@app.route('/api/live_stats')
@is_faculty_logged_in
def api_live_stats():
    return dict(shared_stream.stats)


@app.route('/api/gps', methods=['POST'])
@is_faculty_logged_in
def api_gps():
    """Browser GPS from the live page (Geolocation API) for accurate geofencing."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        lat, lon = float(data['lat']), float(data['lon'])
    except (KeyError, TypeError, ValueError):
        return {"ok": False}, 400
    session['gps'] = {"lat": lat, "lon": lon}
    return {"ok": True, "location": f"{lat:.4f},{lon:.4f}"}




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
