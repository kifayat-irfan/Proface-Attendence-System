"""Shared camera stream processor (Phase 7).

Problem: every /video_feed viewer ran its own cv2.VideoCapture + full
detection loop — 10 viewers = 10 cameras + 10x CPU.

Solution: ONE background thread owns the camera and runs detection,
annotation and attendance marking. Viewers only read the latest JPEG.
CPU stays flat no matter how many faculty watch.
"""
import os
import threading
import time

import cv2
import numpy as np


class SharedStream:
    """Single-producer, multi-consumer MJPEG stream."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._viewers = 0
        self._running = False
        self._thread = None
        self._ctx = {}          # marking context from the faculty session
        self._last_viewer_at = 0
        self.stats = {"marked_this_session": 0, "faces_seen": 0,
                      "viewers": 0, "running": False}

    # ---- called from request threads ----
    def start(self, ctx):
        """Register a viewer; boot the camera thread on first viewer."""
        with self._lock:
            self._viewers += 1
            self._last_viewer_at = time.time()
            self._ctx = dict(ctx)  # latest faculty's lecture context wins
            self.stats["viewers"] = self._viewers
            if not self._running:
                self._running = True
                self.stats["running"] = True
                self.stats["marked_this_session"] = 0
                self._thread = threading.Thread(
                    target=self._loop, daemon=True, name="proface-stream")
                self._thread.start()

    def stop(self):
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
            self.stats["viewers"] = self._viewers
            self._last_viewer_at = time.time()

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    # ---- background thread (needs app hooks injected) ----
    def _loop(self):
        """Camera owner: detect -> mark -> annotate -> publish JPEG.

        App-specific hooks (recognize_faces, detect_faces_and_annotate,
        _mark_one_attendance, get_camera_source, get_user_location) are
        injected via `bind()` so this module stays import-light for tests.
        """
        hooks = self._hooks
        cap = cv2.VideoCapture(hooks['get_camera_source']())
        if not cap.isOpened():
            err = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err, "Camera unavailable - check CAMERA_SOURCE",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', err)
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()
            time.sleep(5)
            with self._lock:
                self._running = False
                self.stats["running"] = False
            return

        # Location once per session
        gps = self._ctx.get('gps')
        if gps:
            location_str = f"{gps['lat']:.4f},{gps['lon']:.4f} (GPS)"
        else:
            loc = hooks['get_user_location']().get('location', 'Unknown')
            location_str = f"{loc[0]},{loc[1]}" if loc != 'Unknown' else 'Unknown'

        marked_rollnos = set()
        idle_since = None
        src_path = hooks['get_camera_source']()
        is_file = isinstance(src_path, str) and os.path.isfile(src_path)
        try:
            while True:
                with self._lock:
                    viewers = self._viewers
                # Auto-stop 60s after the last viewer leaves
                if viewers == 0:
                    idle_since = idle_since or time.time()
                    if time.time() - idle_since > 60:
                        break
                else:
                    idle_since = None

                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        # Demo mode: loop the video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.2)
                    continue
                detections = hooks['recognize_faces'](frame)
                self.stats["faces_seen"] = len(detections)
                frame = hooks['detect_faces_and_annotate'](
                    frame, detections, location_str)
                for det in detections:
                    rollno = det["rollno"]
                    if not rollno or rollno in marked_rollnos:
                        continue
                    if det.get("live"):
                        if hooks['mark_attendance'](rollno, location_str,
                                                    self._ctx):
                            self.stats["marked_this_session"] += 1
                    marked_rollnos.add(rollno)
                _, jpeg = cv2.imencode('.jpg', frame)
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
        finally:
            cap.release()
            with self._lock:
                self._running = False
                self.stats["running"] = False

    def bind(self, **hooks):
        """Inject app functions (keeps this module testable standalone)."""
        self._hooks = hooks


# Process-wide singleton
shared_stream = SharedStream()
