import time
import os
from datetime import datetime
from scipy.spatial import distance as dist
import requests
from keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

from flask_migrate import Migrate
from functools import wraps
import cv2
import dlib
import numpy as np
from geopy.distance import geodesic
import face_recognition


# Paths to cascade files and emotion model
current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_face_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
cascade_eye_path = os.path.join(current_dir,  'haarcascade_eye.xml')
emotion_model_path = os.path.join(current_dir, 'emotion_model.h5')
# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat'))
face_rec_model = dlib.face_recognition_model_v1(os.path.join(current_dir, 'dlib_face_recognition_resnet_model_v1.dat'))
# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cascade_face_path)
eye_cascade = cv2.CascadeClassifier(cascade_eye_path)

# Load emotion recognition model
emotion_model = load_model(emotion_model_path)


# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#################################################################################
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-me')
# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


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
    password = db.Column(db.String(80))
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
    password = db.Column(db.String(80))
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



    with app.app_context():
        
        db.create_all()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/login_student', methods=['GET', 'POST'])
def login_student():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        student = Student.query.filter_by(email=email, password=password).first()

        if student is not None:
            
            
            if not student.approved:  # Prevent unapproved students from signing in
                flash('Approval pending. Please wait for admin approval.', 'warning')
                return render_template('login_student.html')
            else:
                session['std_logged_in'] = True
                uname = email.split('@')[0]
                session['uname'] = uname
                session['roll_no'] = student.rollno

                flash('You are now logged in', 'success')
                return redirect(url_for('student'))
        else:
            flash('Invalid email or password.', 'danger')

        return render_template('login_student.html')

    return render_template('login_student.html')


@app.route('/login_faculty', methods=['GET', 'POST'])
def login_faculty():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        faculty = Faculty.query.filter_by(email=email, password=password).first()

        if faculty is not None:
            # Admin login handling
            if faculty.is_admin:
                session['fty_logged_in'] = True
                session['uname'] = email.split('@')[0]
                session['name'] = faculty.name
                session['is_admin'] = True
                flash('Admin login successful', 'success')
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
            new_student = Student(email=email, password=password, rollno=rollno, name=name)
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
            new_faculty = Faculty(email=email, password=password, name=name, course=course, is_admin=is_admin)
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
                password=request.form['password'],
                pic_path=f'static/images/users/{request.form["rollno"]}.jpg',
                registered_on=datetime.now()
            )
            db.session.add(new_student)
            db.session.commit()

            if os.path.isfile('static/images/users/temp.jpg'):
                os.rename('static/images/users/temp.jpg',
                          f'static/images/users/{request.form["rollno"]}.jpg')
            if 'img_captured' in session:
                session.pop('img_captured')
            flash('Student registration successful', 'success')
            return render_template('register_student.html')
        else:
            flash('Student with this email already exists!', 'danger')

    if os.path.isfile('static/images/users/temp.jpg'):
        temp_pic = True
    else:
        temp_pic = False

    return render_template('register_student.html', temp_pic=temp_pic)


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
                password=request.form['password'],
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
def faculty_list():
    faculty = Faculty.query.all()
    return render_template('faculty_info.html', faculty=faculty)

@app.route('/students_list')
def students_list():
    students = Student.query.all()
    return render_template('student_info.html', students=students)

@app.route('/lectures_list')
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
    db.session.delete(student)
    db.session.commit()
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
        student.password = request.form['password']
        db.session.commit()
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
        flash('Attendance record updated successfully', 'success')
        return redirect(url_for('view_lectures_attendance'))

    return render_template('edit_attendance.html', attendance=attendance)

@app.route('/delete_attendance/<int:att_id>', methods=['POST'])
@is_faculty_logged_in
def delete_attendance(att_id):
    attendance = Attendance.query.get_or_404(att_id)
    db.session.delete(attendance)
    db.session.commit()
    flash('Attendance record deleted successfully', 'success')
    return redirect(url_for('view_lectures_attendance'))


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route("/capture_image")
@is_faculty_logged_in
def capture_image():
    session['dt'] = datetime.now()
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

    session['img_captured'] = True

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


@app.route("/logout")
def logout():
    if 'std_logged_in' or 'fty_logged_in' in session:
        session.clear()
    return redirect(url_for('index'))

# Emotion detection function
def detect_emotion(image):
    # Preprocess image as required by the emotion detection model
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(gray_image, (48, 48))
    face_array = np.expand_dims(face_resized, axis=0)
    face_array = np.expand_dims(face_array, axis=-1) / 255.0
    
    # Predict using the emotion model
    prediction = emotion_model.predict(face_array)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    return emotion




def detect_faces(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    return faces

def get_face_encodings(image, shape_predictor, face_recognition_model):
    faces = detect_faces(image)
    encodings = []
    for face in faces:
        shape = shape_predictor(image, face)
        encoding = np.array(face_recognition_model.compute_face_descriptor(image, shape))
        encodings.append(encoding)
    return encodings

def compare_faces(known_encodings, face_encoding):
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    match = np.any(distances <= 0.6)
    return match, distances

def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = get_face_encodings(rgb_frame, shape_predictor, face_recognition_model)
    
    recognized_rollnos = []
    for face_encoding in face_encodings:
        match, distances = compare_faces(np.array(known_face_encodings), face_encoding)
        if match:
            best_match_index = np.argmin(distances)
            rollno = known_face_rollnos[best_match_index]
            recognized_rollnos.append(rollno)
    
    return recognized_rollnos

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
def detect_faces_and_annotate(frame, rollnos, location, emotions):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(rgb_frame)
    
    # Annotate each detected face
    for (i, face) in enumerate(faces):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        rollno = rollnos[i] if i < len(rollnos) else "Unknown"
        emotion = emotions[i] if i < len(emotions) else "Unknown"
        
        # Get shape for liveness detection
        shape = shape_predictor(rgb_frame, face)
        liveness = detect_liveness(shape)
        liveness_status = "Live" if liveness else "Fake"
        # Lookup student name based on roll number
        student_name = "Unknown Student"
        if rollno != "Unknown":
            student = Student.query.filter_by(rollno=rollno).first()
            if student:
                student_name = student.name
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Annotate roll number, emotion, and liveness
        cv2.putText(frame, f'Name: {student_name}', (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Roll No: {rollno}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Liveness: {liveness_status}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the location text in the upper left corner
    cv2.putText(frame, f'Location: {location}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return frame
# Load known faces and their encodings
known_face_encodings = []
known_face_rollnos = []

# Initialize dlib's face detector, shape predictor, and face recognition model
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Example: Load known faces from a directory called 'known_faces'
known_faces_dir = "static/images/users"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = dlib.load_rgb_image(image_path)
        encodings = get_face_encodings(image, shape_predictor, face_recognition_model)
        if encodings:
            known_face_encodings.append(encodings[0])
            rollno = filename.split(".")[0]  # Assuming filename is rollno.jpg or rollno.png
            known_face_rollnos.append(rollno)

@app.route('/fr_attendance')
def mark_face_attendance():
    # Open the camera
    cap = None
    for camera_index in range(10):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Using camera index {camera_index}")
            break
    else:
        print("No valid camera found.")
        return "No valid camera found", 400

    if cap is None or not cap.isOpened():
        return "Failed to open camera", 500

    # Set to track marked roll numbers
    marked_rollnos = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Get user location
        location_data = get_user_location()
        location = location_data.get('location', 'Unknown')
        rollnos = recognize_face(frame)
        emotions = [detect_emotion(frame) for _ in rollnos]

        # Convert location tuple to string
        location_str = f"{location[0]},{location[1]}" if location != 'Unknown' else 'Unknown'

        if rollnos:
            for rollno in rollnos:
                if rollno not in marked_rollnos:
                    # Lookup student name based on roll number
                    student = Student.query.filter_by(rollno=rollno).first()
                    student_name = student.name if student else 'Unknown Student'
                    
                    # Get the current lecture number and course from the session or database
                    lecture_number = session.get('lecture_no', 'Unknown Lecture')
                    course = session.get('course', 'Unknown')
                    marked_by = session.get('name', 'Unknown')

                    # Check if attendance already marked for this rollno, lecture_no, and marked_date
                    existing_attendance = Attendance.query.filter_by(
                        rollno=rollno,
                        lecture_no=lecture_number,
                        marked_date=datetime.now().date()
                    ).first()

                    if not existing_attendance:
                        # Mark attendance
                        attendance = Attendance(
                            name=student_name,
                            rollno=rollno,
                            course=course,
                            lecture_no=lecture_number,
                            marked_by=marked_by,
                            marked_date=datetime.now().date(),
                            marked_time=datetime.now().time(),
                            location=location_str
                        )
                        db.session.add(attendance)
                        db.session.commit()
                        flash(f"Attendance marked for roll number {rollno}", 'success')

                        # Add roll number to the set to prevent re-marking
                        marked_rollnos.add(rollno)
                    else:
                        flash(f"Roll number {rollno} has already been marked for this lecture.", 'info')

        # Display the frame with annotations
        processed_frame = detect_faces_and_annotate(frame, rollnos, location, emotions)
        if processed_frame is not None:
            cv2.imshow('Facial Recognition with Liveness and Emotion Detection', processed_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return redirect(url_for('mark_attendance_2'))


if __name__ == '__main__':
       # Initialize database tables
     app.run(debug=True)
