import threading
import cv2
import face_recognition
import numpy as np
from flask import session
import os
from datetime import datetime
from models import db, Attendance  # Adjust the import as needed

class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super(VideoCaptureThread, self).__init__()
        self.video_capture = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True
        self.flag = False
        self.load_known_faces()
        self.running = True

    def load_known_faces(self):
        known_faces_filenames = []
        for (dirpath, dirnames, filenames) in os.walk('static/images/users'):
            known_faces_filenames.extend(filenames)
            break

        for filename in known_faces_filenames:
            face = face_recognition.load_image_file('static/images/users/' + filename)
            encodings = face_recognition.face_encodings(face)
            if encodings:
                self.known_face_names.append(filename[:-4])
                self.known_face_encodings.append(encodings[0])

    def run(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if self.process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    roll_name = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        roll_name = self.known_face_names[best_match_index]

                        roll = roll_name.split('-')[0]
                        exists = Attendance.query.filter_by(rollno=roll).first()
                        if exists:
                            is_marked_already = Attendance.query.filter(
                                Attendance.rollno == roll,
                                Attendance.course == session['course'],
                                Attendance.lecture_no == int(session['lecture_no'])
                            ).all()

                        if exists is None:
                            attendance = Attendance(
                                rollno=roll,
                                course=session['course'],
                                lecture_no=session['lecture_no'],
                                marked_by=session['name'],
                                marked_date=datetime.date(datetime.now()),
                                marked_time=datetime.time(datetime.now())
                            )
                            db.session.add(attendance)
                            db.session.commit()
                            self.flag = True

                        elif len(is_marked_already) == 0:
                            attendance = Attendance(
                                rollno=roll,
                                course=session['course'],
                                lecture_no=session['lecture_no'],
                                marked_by=session['name'],
                                marked_date=datetime.date(datetime.now()),
                                marked_time=datetime.time(datetime.now())
                            )
                            db.session.add(attendance)
                            db.session.commit()
                            self.flag = True

                    face_names.append(roll_name)

            self.process_this_frame = not self.process_this_frame

            for (top, right, bottom, left), roll_name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, roll_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                if self.flag:
                    cv2.putText(frame, 'Marked', (left + 12, bottom - 12), font, 0.5, (255, 255, 255), 1)

            cv2.imshow('Marking attendance', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
