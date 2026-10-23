# 🚀 ProFace Deployment Guide

## Recommended stack (free, portfolio-ready)

| Layer | Service | Why | Cost |
|---|---|---|---|
| **App hosting** | [Hugging Face Spaces](https://huggingface.co/spaces) (Docker) | Free, 16 GB RAM (dlib needs it!), live URL for portfolio | **Free** |
| **Database** | [Supabase](https://supabase.com) (PostgreSQL) | Free managed Postgres, daily auto-backups, connection pooling | **Free** |
| **Photo storage** | Local `static/images/users/` (or Supabase Storage later) | Works out of the box | Free |

### Why NOT Vercel?
Vercel is serverless: functions time out, no persistent filesystem, no
long-running camera streams, and dlib (native C++, ~1 GB build) doesn't fit.
ProFace needs a real server — Docker on HF Spaces is the right call.

### Why NOT Supabase alone?
Supabase is a database/auth/storage backend — it doesn't run Python Flask
apps. Use it for what it's great at (PostgreSQL), and run the app on Spaces.

---

## Step 1: Supabase database (10 min)

1. Create a project at [supabase.com](https://supabase.com) → **Project Settings → Database**
2. Copy the **Connection string** (URI). It looks like:
   `postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres`
3. For Spaces, use the **pooled** connection (port 6543) — add `?pgbouncer=true`:
   `postgresql://postgres:[password]@db.[ref].supabase.co:6543/postgres?pgbouncer=true`
4. Supabase runs **daily automatic backups** — the manual export at
   `/admin/backup` is your extra safety net.

## Step 2: Hugging Face Space (15 min)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   → choose the **Docker** SDK.
2. Push this repo to the Space (or connect the GitHub repo).
3. In the Space **Settings → Variables**, add secrets:
   - `DATABASE_URL` = your Supabase pooled connection string
   - `SECRET_KEY` = any long random string
   - `INSTITUTION_NAME` = e.g. `Khushal Khan Khattak University`
   - `CAMERA_SOURCE` = leave `0` for on-premise, or point at a demo video
     (see below) for the public portfolio demo.
4. The Dockerfile runs migrations automatically and serves on port 7860.

> **First build takes ~20 min** (dlib compiles from source). Later builds are cached.

## Step 3: Demo video mode (portfolio)

Cloud servers have no webcam. For the **public live demo**, use a sample video
as the camera — the stream loops it automatically:

```
CAMERA_SOURCE=/app/demo/classroom.mp4
```

Add any MP4 of people facing the camera to `demo/` and set the env var.
Face detection, liveness boxes, emotion labels and attendance marking all work
on video — visitors see the real system in action.

For a **real university deployment**, run the same Docker image on-premise
(a PC in the classroom) with `CAMERA_SOURCE=0` (webcam) or an RTSP URL:

```
CAMERA_SOURCE=rtsp://user:pass@192.168.1.10:554/stream
```

## Step 4 (optional): custom domain

HF Spaces supports custom domains on paid plans; otherwise the
`https://<you>-proface.hf.space` URL is perfect for a CV/portfolio.

---

## Environment variables cheat sheet

| Variable | Example | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://...` | Supabase Postgres (default: local SQLite) |
| `SECRET_KEY` | random 32+ chars | Flask sessions — **always change in prod** |
| `INSTITUTION_NAME` | `KKK University` | Branding on every page |
| `CAMERA_SOURCE` | `0`, RTSP URL, or `/app/demo/x.mp4` | Camera |
| `RECOGNITION_ENGINE` | `dlib` (default) / `arcface` | Face engine |
| `PORT` | `7860` | Served port (HF Spaces needs 7860) |

## Health check

`GET /api/v1/health` → `{"status":"ok", ...}` — point uptime monitors here.
