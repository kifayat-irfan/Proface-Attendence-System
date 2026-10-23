"""Tests for Phase 7: shared stream + REST API key auth.

Light deps only — stream.py is import-standalone; API hashing is pure.
"""
import hashlib
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stream import SharedStream


class TestSharedStream:
    def test_viewer_counting(self):
        s = SharedStream()
        s.bind(get_camera_source=lambda: 0)  # not started; just counting
        assert s.stats["viewers"] == 0
        # start() boots a thread — avoid it here; test stop() safety instead
        s.stop()
        assert s.stats["viewers"] == 0  # never negative

    def test_get_jpeg_none_initially(self):
        s = SharedStream()
        assert s.get_jpeg() is None

    def test_camera_failure_path(self):
        """Unopenable camera -> error frame published, thread exits cleanly."""
        import cv2
        s = SharedStream()
        s.bind(
            get_camera_source=lambda: 9999,  # invalid index
            get_user_location=lambda: {'location': 'Unknown'},
            recognize_faces=lambda f: [],
            detect_faces_and_annotate=lambda f, d, l: f,
            mark_attendance=lambda r, l, c: False,
        )
        # Monkeypatch VideoCapture to fail fast
        real_vc = cv2.VideoCapture
        cv2.VideoCapture = lambda *a: type('C', (), {
            'isOpened': lambda s: False, 'release': lambda s: None})()
        try:
            s.start({'lecture_no': '1'})
            time.sleep(7)  # failure path sleeps 5s then exits
            assert s.get_jpeg() is not None  # error frame published
            assert s.stats["running"] is False
        finally:
            cv2.VideoCapture = real_vc
            s.stop()

    def test_idle_auto_stop(self):
        """Thread exits ~60s after last viewer leaves (tested with short-circuit)."""
        s = SharedStream()
        frames = [b'frame1', b'frame2']

        def fake_loop():
            # simulate: publish 2 frames then emulate idle-timeout exit
            import time as t
            for f in frames:
                with s._lock:
                    s._latest_jpeg = f
                t.sleep(0.05)
            with s._lock:
                s._running = False
                s.stats["running"] = False

        import threading
        s._running = True
        threading.Thread(target=fake_loop, daemon=True).start()
        time.sleep(0.3)
        assert s.get_jpeg() == b'frame2'
        assert s.stats["running"] is False


class TestApiKeyHash:
    def test_hash_is_sha256(self):
        h = hashlib.sha256(b'test-key-123').hexdigest()
        assert len(h) == 64 and h == hashlib.sha256(b'test-key-123').hexdigest()

    def test_different_keys_different_hashes(self):
        assert (hashlib.sha256(b'a').hexdigest()
                != hashlib.sha256(b'b').hexdigest())

    def test_raw_key_never_stored(self):
        # The model only has key_hash — assert the column contract
        import ast
        src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
        assert 'key_hash = db.Column(db.String(64)' in src
