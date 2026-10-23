---
title: ProFace — AI Attendance System
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ProFace — AI-Powered Facial Recognition Attendance System

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Flask](https://img.shields.io/badge/Flask-webapp-lightgrey)
![dlib](https://img.shields.io/badge/dlib-face%20recognition-orange)
![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)

A complete, production-grade **facial recognition attendance system** built with Flask — evolved from my university Final Year Project (2024) into a full-featured platform. Students enroll with photos; faculty mark attendance in real time through **face recognition**, with **dual liveness anti-spoofing**, **emotion detection**, **GPS geofencing**, **analytics**, **smart alerts**, and a **REST API**.

![ProFace demo](static/Documentation.gif)

## How It Works

```
  Camera frame (webcam / IP camera / video file)
      │
      ▼
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ Face detect │──▶│ 68-landmark align │──▶│ Face embedding      │
│ (dlib HOG)  │   │ (shape predictor) │   │ dlib 128-d OR       │
└─────────────┘   └──────────────────┘   │ ArcFace 512-d       │
                                         └─────────┬───────────┘
                                                   │ distance match
                                                   ▼
                                          ┌────────────────┐
                                          │ Match → roll no │
                                          └───────┬────────┘
                                                  ▼
                        ┌─────────────────────────────────────────┐
                        │ Liveness: anti-spoofing CNN + EAR blink  │
                        │ Emotion: Keras CNN (7 emotions)          │
                        │ Location: IP/GPS geofencing vs campus    │
                        └───────────────────┬─────────────────────┘
                                            ▼
                    ┌─────────────────────────────────┐
                    │ Shared stream → MJPEG to browser │
                    │ Attendance recorded (unique per  │
                    │ student/lecture/day)             │
                    └─────────────────────────────────┘
```

One camera/detection thread serves **all** browser viewers (shared stream); per-viewer MJPEG relay keeps CPU flat as viewers scale.

## Key Features

**AI & Computer Vision**
- **Face recognition attendance** — dlib HOG + 68-point landmarks + ResNet-29 (128-d, Euclidean ≤ 0.6), or **ArcFace** via insightface (512-d, cosine ≥ 0.5, `RECOGNITION_ENGINE=arcface`). Detection runs on a downscaled frame with a single detection pass shared between recognition and overlay
- **Multi-angle enrollment** — front/left/right/up/glasses templates per student with a **photo quality gate** (blur, brightness, face-size checks); matcher tries every angle
- **Dual liveness anti-spoofing** — trained CNN (`liveness_model.h5`) scores every face crop **plus** eye-aspect-ratio blink detection; only live faces are marked, printed photos/screens are rejected
- **Emotion detection** — Keras CNN classifies 7 emotions per face crop
- **Geofencing** — IP-based location plus browser GPS, checked against campus coordinates

**Web Application**
- **Three roles** — Student, Faculty, Admin dashboards (dark-navy sidebar UI)
- **Admin approval workflow** — signups stay pending until approved
- **Live browser streaming** — `/live` MJPEG page, no desktop window needed
- **Multi-camera / IP support** — `CAMERA_SOURCE` env (device index, RTSP URL, or video file for demos)
- **PWA** — installable, responsive, works on phones/tablets
- **📈 Analytics** — KPI cards, 14-day trend, per-lecture breakdown, **defaulter list** (< 75%) with one-click **parent email alerts**
- **🔔 Smart alerts** — absence streaks (3+ days), low attendance, and face-aging re-enrollment reminders (idempotent, throttled)
- **📋 Audit log** — every sensitive action recorded, admin-viewable
- **💾 Backup** — one-click full JSON export (excludes secrets/embeddings)
- **🏫 Multi-institution** — `INSTITUTION_NAME` branding + `TENANT_SLUG`; run side-by-side via `docker-compose.multi.yml`

**Security**
- **scrypt** password hashing (legacy plaintext auto-migrated on first login)
- Rate limiting (10/min on auth), HttpOnly/SameSite cookies, 8-hour sessions
- **REST API v1** (`/api/v1/...`) with revocable SHA-256 **API keys** (`X-API-Key` header)

## Tech Stack

| Layer | Technology |
|---|---|
| Web | Flask, Jinja2, Tailwind, Chart.js, PWA (service worker + manifest) |
| Database | SQLite (dev) / PostgreSQL via `DATABASE_URL` + Flask-Migrate (Alembic) |
| Face recognition | dlib (HOG + 68 landmarks + ResNet-29) or ArcFace/insightface |
| Emotion / Liveness | Keras CNNs (`emotion_model.h5`, `liveness_model.h5`) + EAR blink |
| Location | geopy + ipinfo.io + browser Geolocation API |
| Security | scrypt, Flask-Limiter, API keys |
| Alerts | SMTP email to parents (env-configured) |
| Deploy | Docker + docker-compose, gunicorn |

## Project Structure

```
Proface-Attendence-System/
├── app.py                  # Main Flask app: routes, recognition pipeline, API v1
├── security.py             # scrypt hashing, auth decorators, liveness policy
├── recognition.py          # DlibEngine + ArcFaceEngine (pluggable)
├── quality.py              # Enrollment photo quality gate
├── stream.py               # SharedStream: one camera thread for all viewers
├── analytics.py            # KPIs, trends, defaulter list
├── alerts.py               # Smart alerts (streaks, low attendance, re-enrollment)
├── notify.py               # Parent email alerts (SMTP)
├── add_admin.py            # Secure CLI: create an admin
├── download_models.py      # Fallback: re-fetch shape predictor from dlib.net
├── templates/              # Jinja2 templates (dashboards, auth, analytics, ...)
├── static/                 # CSS/JS, PWA manifest, logo
├── migrations/             # Alembic migrations (9 total)
├── emotion_model.h5 / liveness_model.h5   # Trained CNNs
├── dlib_face_recognition_resnet_model_v1.dat
├── shape_predictor_68_face_landmarks.dat  # 95MB, ships in repo
├── requirements.txt / Dockerfile / docker-compose.yml
├── DEPLOY.md               # Deployment guide
└── tests/                  # 56 tests
```

## Setup

### Prerequisites

- Python 3.11+, a webcam (for live marking)
- **dlib needs a C++ compiler + CMake.** On Windows, install [CMake](https://cmake.org/download/) and Visual Studio Build Tools (C++ workload) *before* `pip install`.

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
- `shape_predictor_68_face_landmarks.dat` (95MB)
- `dlib_face_recognition_resnet_model_v1.dat` (22MB)
- `emotion_model.h5` / `liveness_model.h5`

### 3. Configure

```bash
export SECRET_KEY="your-secret-key"
export DATABASE_URL="postgresql://user:pass@host:5432/proface"  # optional; SQLite by default
export RECOGNITION_ENGINE="arcface"   # optional; dlib is default
export CAMERA_SOURCE="0"              # or RTSP URL, or video file path
export INSTITUTION_NAME="My University"
```

### 4. Migrate & run

```bash
flask --app app db upgrade
python app.py
```

Open `http://127.0.0.1:5000`.

### 5. Docker (one command)

```bash
docker compose up --build
```

### 6. First admin

```bash
python add_admin.py
```

## API v1

```
GET  /api/v1/health              # no auth: status, version, tenant
GET  /api/v1/students            # X-API-Key required
GET  /api/v1/students/<rollno>
GET  /api/v1/attendance
POST /api/v1/attendance          # {rollno, lecture_no, course}
GET  /api/v1/analytics/summary
GET  /api/v1/alerts
```

Manage keys at `/api_keys` (admin). Keys are shown once and stored as SHA-256 hashes.

## Notes & Limitations

- **Liveness CNN class index** (`LIVENESS_REAL_INDEX = 1`) is a documented assumption (alphabetical: fake=0, real=1). Verify once with a physical camera test (real face + printed photo); flip to `0` if inverted on your model copy.
- Embeddings are cached in the DB per engine; switching `RECOGNITION_ENGINE` re-enrolls automatically.
- Geolocation via IP is approximate — use browser GPS share for precise geofencing.
- The default campus coordinates point to Khushal Khan Khattak University, Karak — change `KARAK_LOCATION` in `app.py` for your own campus.

## License

MIT — see [LICENSE](LICENSE).
