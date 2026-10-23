# ProFace — AI-Powered Facial Recognition Attendance System

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Flask](https://img.shields.io/badge/Flask-webapp-lightgrey)
![dlib](https://img.shields.io/badge/dlib-face%20recognition-orange)

A complete, production-style **facial recognition attendance system** built with Flask — my university Final Year Project. Students register with a photo; faculty mark attendance in real time through **face recognition**, with **emotion detection**, **blink-based liveness anti-spoofing**, and **GPS geofencing** so attendance can only be marked near campus.

![ProFace demo](static/Documentation.gif)

## How It Works

```
  Camera frame
      │
      ▼
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ Face detect │──▶│ 68-landmark align │──▶│ ResNet-29 face      │
│ (dlib HOG)  │   │ (shape predictor) │   │ embedding (128-d)   │
└─────────────┘   └──────────────────┘   └─────────┬───────────┘
                                                   │ Euclidean
                                                   │ distance ≤ 0.6
                                                   ▼
                                          ┌────────────────┐
                                          │ Match → roll no │
                                          └───────┬────────┘
                                                  ▼
                        ┌─────────────────────────────────────────┐
                        │ Liveness: eye-aspect-ratio blink check   │
                        │ Emotion: Keras CNN (7 emotions)          │
                        │ Location: IP geolocation vs campus GPS   │
                        └───────────────────┬─────────────────────┘
                                            ▼
                                   Attendance recorded
                              (unique per student/lecture/day)
```

## Key Features

**AI & Computer Vision**
- **Face recognition attendance** — dlib HOG detector + 68-point landmarks + ResNet-29 (128-d embeddings); matches against registered student photos. Detection runs on a downscaled frame (~4x faster) with a single detection pass shared between recognition and the on-screen overlay
- **Emotion detection** — Keras CNN classifies 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) on each detected face crop
- **Dual liveness anti-spoofing** — a trained anti-spoofing CNN (`liveness_model.h5`) scores every face crop *plus* eye-aspect-ratio (EAR) blink detection; a face is marked only if it passes the combined check, so printed photos and replayed videos are rejected
- **Geofencing** — IP-based location (fetched once per marking session, not per frame) checked against campus coordinates; attendance records carry the marking location

**Web Application**
- **Three roles** — Student, Faculty, and Admin panels, each with its own dashboard
- **Admin approval workflow** — signups stay pending until an admin approves them
- **Registration with photo capture** — capture the student's face from webcam during registration; photo becomes their recognition template
- **Two marking modes** — AI face-recognition marking *and* manual fallback
- **Dashboards & reports** — per-course attendance views, lecture lists, student/faculty management
- **CSV export** — download attendance per lecture (faculty) or per course (student)
- **Full CRUD** — edit/delete students and attendance records
- **📈 Analytics dashboard** (`/analytics`) — KPI cards, 14-day attendance trend chart, per-lecture breakdown, automatic **defaulter list** (< 75%) with one-click **parent email alerts**

## Tech Stack

| Layer | Technology |
|---|---|
| Web | Flask, Jinja2, Bootstrap 4 / Tailwind, Chart.js, JavaScript |
| Database | SQLite + SQLAlchemy + Flask-Migrate (Alembic) |
| Face recognition | dlib (HOG + 68 landmarks + ResNet-29), `face_recognition` |
| Emotion | Keras CNN (`emotion_model.h5`) |
| Liveness | Eye-aspect-ratio blink detection (OpenCV) + anti-spoofing CNN (`liveness_model.h5`) |
| Location | geopy + ipinfo.io geolocation |
| Alerts | SMTP email to parents (env-configured, no credentials in code) |
| Deploy | Docker + docker-compose, gunicorn |

## Project Structure

```
Proface-Attendence-System/
├── app.py                  # Main Flask app: routes, recognition pipeline, liveness, geofencing
├── models.py               # SQLAlchemy models (Student, Faculty, Attendance)
├── video_thread.py         # Threaded video-capture attendance worker
├── add_admin.py             # CLI helper: create an admin from the Python console
├── download_models.py       # Downloads the dlib shape predictor (~95MB)
├── templates/              # 20+ Jinja2 templates (dashboards, auth, attendance views)
├── static/                 # CSS/JS, Bootstrap, sample user photos
├── migrations/             # Alembic migrations (approval flow, unique constraints, ...)
├── emotion_model.h5        # Trained Keras emotion classifier (7 classes)
├── liveness_model.h5       # Trained liveness model
├── dlib_face_recognition_resnet_model_v1.dat  # dlib 128-d face embedding model
├── haarcascade_*.xml       # OpenCV Haar cascades (eye + frontal face)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- A webcam (for registration photo capture and live marking)
- **dlib needs a C++ compiler + CMake to build.** On Windows, install [CMake](https://cmake.org/download/) and Visual Studio Build Tools (C++ workload) *before* `pip install`. This is the only tricky dependency — everything else is pure pip.

### 1. Install

```bash
git clone https://github.com/kifayat-irfan/Proface-Attendence-System.git
cd Proface-Attendence-System
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Models — all included! ✅

All AI models ship in the repo (verified authentic):
- `shape_predictor_68_face_landmarks.dat` (95MB) — 68-point facial landmarks
- `dlib_face_recognition_resnet_model_v1.dat` (22MB) — 128-d face embeddings
- `emotion_model.h5` / `liveness_model.h5` — trained CNNs

(`download_models.py` remains as a fallback to re-fetch the shape predictor from dlib.net if ever needed.)

### 3. Configure (optional)

```bash
# Flask secret key — set your own in production
export SECRET_KEY="your-secret-key"      # Windows: set SECRET_KEY=your-secret-key
```

### 4. Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`. The SQLite database is created automatically on first run.

### 5b. One-command Docker run (alternative)

```bash
docker compose up --build
```

Then open `http://localhost:5000`. The DB and registered face photos persist in `./instance` and `./static/images/users`. Set `SECRET_KEY` (and `SMTP_*` for parent alerts) in the environment or a `.env` file.

> First build takes a while — dlib compiles from source. Live camera marking needs a webcam + display on the host (see `docker-compose.yml` notes); dashboards, analytics, CSVs and the API work headless.

### 6. Parent absence alerts (optional)

1. Add each student's **parent email** on the Register/Edit Student page.
2. Set SMTP via environment (Gmail: use an App Password, never your real password):
   ```bash
   export SMTP_HOST=smtp.gmail.com SMTP_PORT=587
   export SMTP_USER=you@gmail.com SMTP_PASSWORD=your-app-password
   ```
3. Open **Analytics → 📧 Alert parents** — one click emails every defaulter's parent.

### 5. First admin

```bash
python add_admin.py     # from the Python console, as noted in the file
```

## Usage Walkthrough

1. **Admin** approves pending student/faculty signups (`/admin/approve_requests`)
2. **Faculty** registers students with a webcam photo capture (`/student_registration`) — the photo becomes the face template stored under `static/images/users/<rollno>.jpg`
3. **Faculty** starts a lecture (`/attendance`), then opens face marking (`/attendance/fr`) — the camera recognizes students, checks blink liveness + emotion + location, and records attendance (one record per student/lecture/day, enforced by a DB unique constraint)
4. **Students** log in to view their own attendance per course and download it as CSV
5. **Everyone** gets role-appropriate dashboards; faculty can edit/delete records and export lecture CSVs

## Notes & Limitations

- Passwords are stored in plaintext in this academic build — **hash them (e.g. Werkzeug/bcrypt) before any real deployment.**
- Face templates are the registered JPEGs themselves; for larger deployments, pre-computing and caching embeddings is the natural next step.
- Geolocation uses IP-based lookup (`ipinfo.io`), which is approximate — fine for campus-level geofencing, not for precise positioning.
- The default campus coordinates point to Khushal Khan Khattak University, Karak — change `KARAK_LOCATION` in `app.py` for your own campus.

## License

MIT — see [LICENSE](LICENSE).
