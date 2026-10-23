"""Tests for Phase 6: photo quality assessment + multi-angle templates.

Light deps only — uses a fake engine (no dlib/insightface needed).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quality import assess_photo, issue_text, ISSUE_MESSAGES


class FakeEngine:
    """Pretends one face fills the center of any image 200px+ wide."""
    name = 'fake'
    dim = 128

    def detect_and_encode(self, bgr_image):
        h, w = bgr_image.shape[:2]
        if w < 200:
            return []  # too small -> no face
        return [((w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                 np.zeros(128, dtype=np.float32))]


def _good_image():
    # 640x480, mid-bright, textured (high contrast, sharp edges)
    rng = np.random.default_rng(7)
    img = rng.integers(90, 170, (480, 640, 3), dtype=np.uint8)
    cv2 = pytest.importorskip('cv2')
    cv2.rectangle(img, (160, 120), (480, 360), (200, 200, 200), -1)
    cv2.rectangle(img, (200, 160), (440, 320), (60, 60, 60), -1)
    return img


class TestQuality:
    def test_good_photo_passes(self):
        cv2 = pytest.importorskip('cv2')
        r = assess_photo(_good_image(), FakeEngine())
        assert r['passed'] and r['score'] == 100, r

    def test_no_face_fails(self):
        cv2 = pytest.importorskip('cv2')
        tiny = np.zeros((100, 100, 3), dtype=np.uint8)
        r = assess_photo(tiny, FakeEngine())
        assert not r['passed'] and 'no_face' in r['issues']

    def test_blurry_fails(self):
        cv2 = pytest.importorskip('cv2')
        img = _good_image()
        blurry = cv2.GaussianBlur(img, (51, 51), 0)
        r = assess_photo(blurry, FakeEngine())
        assert not r['passed'] and 'too_blurry' in r['issues']

    def test_dark_fails(self):
        cv2 = pytest.importorskip('cv2')
        img = (_good_image() * 0.1).astype(np.uint8)
        r = assess_photo(img, FakeEngine())
        assert not r['passed'] and 'too_dark' in r['issues']

    def test_score_in_range(self):
        cv2 = pytest.importorskip('cv2')
        r = assess_photo(_good_image(), FakeEngine())
        assert 0 <= r['score'] <= 100


class TestIssueMessages:
    def test_all_issues_have_messages(self):
        for code in ['no_face', 'multiple_faces', 'too_blurry', 'too_dark',
                     'overexposed', 'low_contrast', 'face_too_small']:
            assert code in ISSUE_MESSAGES and ISSUE_MESSAGES[code]

    def test_issue_text_readable(self):
        t = issue_text(['too_blurry', 'too_dark'])
        assert 'blurry' in t and 'dark' in t


class TestMultiAngleFilenames:
    def test_angle_parsing(self):
        for fname, want_roll, want_angle in [
            ('123__front.jpg', '123', 'front'),
            ('123__left.jpg', '123', 'left'),
            ('123.jpg', '123', 'front'),
        ]:
            base = os.path.splitext(fname)[0]
            if '__' in base:
                roll, angle = base.rsplit('__', 1)
            else:
                roll, angle = base, 'front'
            assert (roll, angle) == (want_roll, want_angle)
