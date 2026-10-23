"""Photo quality assessment for face enrollment (Phase 6).

A bad enrollment photo (blurry, dark, tiny face, multiple people) silently
degrades recognition accuracy for that student forever. This module gates
enrollment: photos must pass before their embeddings enter the registry.
"""
import cv2
import numpy as np

# Thresholds tuned on real ProFace webcam captures (640x480):
# real photos score 13-150 (median ~50); heavily blurred photos score <4.
MIN_LAPLACIAN_VAR = 8.0       # below = too blurry
MIN_BRIGHTNESS = 35.0         # mean gray below = too dark
MAX_BRIGHTNESS = 225.0        # mean gray above = overexposed
MIN_CONTRAST = 25.0           # std of gray below = flat/washed out
MIN_FACE_WIDTH_RATIO = 0.12   # face narrower than 12% of frame = too far


def assess_photo(bgr_image, engine):
    """Assess one BGR enrollment photo.

    Returns dict: {passed, score (0-100), issues[], face_rect, detail{}}.
    `engine` is the active recognition engine (for face detection).
    """
    issues = []
    detail = {}
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # --- 1. Face detection: exactly one face ---
    hits = engine.detect_and_encode(bgr_image)
    detail['faces_found'] = len(hits)
    if len(hits) == 0:
        issues.append('no_face')
    elif len(hits) > 1:
        issues.append('multiple_faces')
    face_rect = hits[0][0] if hits else None

    # --- 2. Blur (Laplacian variance on the face region, else whole frame) ---
    if face_rect:
        x1, y1, x2, y2 = face_rect
        roi = gray[max(0, y1):y2, max(0, x1):x2]
    else:
        roi = gray
    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
    detail['laplacian_var'] = round(lap_var, 1)
    if lap_var < MIN_LAPLACIAN_VAR:
        issues.append('too_blurry')

    # --- 3. Brightness / exposure ---
    brightness = float(np.mean(gray))
    detail['brightness'] = round(brightness, 1)
    if brightness < MIN_BRIGHTNESS:
        issues.append('too_dark')
    elif brightness > MAX_BRIGHTNESS:
        issues.append('overexposed')

    # --- 4. Contrast ---
    contrast = float(np.std(gray))
    detail['contrast'] = round(contrast, 1)
    if contrast < MIN_CONTRAST:
        issues.append('low_contrast')

    # --- 5. Face size ---
    if face_rect:
        fw = (x2 - x1) / w
        detail['face_width_ratio'] = round(fw, 3)
        if fw < MIN_FACE_WIDTH_RATIO:
            issues.append('face_too_small')

    # --- Score: start at 100, deduct per issue ---
    deductions = {'no_face': 100, 'multiple_faces': 60, 'too_blurry': 30,
                  'too_dark': 25, 'overexposed': 25, 'low_contrast': 15,
                  'face_too_small': 30}
    score = max(0, 100 - sum(deductions.get(i, 10) for i in issues))
    passed = not issues

    return {'passed': passed, 'score': score, 'issues': issues,
            'face_rect': face_rect, 'detail': detail}


ISSUE_MESSAGES = {
    'no_face': 'No face detected — look directly at the camera.',
    'multiple_faces': 'Multiple faces detected — only the student should be in frame.',
    'too_blurry': 'Photo is too blurry — hold still and try again.',
    'too_dark': 'Too dark — move to better lighting.',
    'overexposed': 'Too bright/washed out — avoid direct light behind you.',
    'low_contrast': 'Low contrast — improve the lighting.',
    'face_too_small': 'Face too small — move closer to the camera.',
}


def issue_text(issues):
    """Human-readable feedback for a list of issue codes."""
    return ' '.join(ISSUE_MESSAGES.get(i, i) for i in issues)
