# ProFace — one-command deploy
# Build:  docker compose up --build
# (First build takes a while: dlib compiles from source.)
FROM python:3.11-slim

# System build deps for dlib + runtime libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Fetch the 95MB dlib shape predictor (too large for git)
RUN python download_models.py

ENV SECRET_KEY=change-me-in-production \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Apply migrations, then serve with gunicorn
CMD ["sh", "-c", "flask --app app db upgrade 2>/dev/null || true; gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 'app:app'"]
