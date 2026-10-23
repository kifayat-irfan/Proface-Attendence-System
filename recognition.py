"""Pluggable face recognition engines for ProFace.

Two engines, one interface — switch with the RECOGNITION_ENGINE env var:

    RECOGNITION_ENGINE=dlib     (default) 128-d, euclidean <= 0.6
    RECOGNITION_ENGINE=arcface             512-d, cosine  >= 0.5  (insightface)

dlib works everywhere with zero extra downloads. ArcFace (InsightFace
buffalo_s) is significantly more accurate across pose, lighting and age —
it downloads ~120MB of models on first use.
"""
import os

import numpy as np


class RecognitionEngine:
    name = "base"
    dim = 0

    def detect_and_encode(self, bgr_frame):
        """Return [(rect(x1,y1,x2,y2), embedding), ...] for each face."""
        raise NotImplementedError

    def match(self, embedding, known):
        """known: list of (rollno, embedding). Return best rollno or None."""
        raise NotImplementedError


class DlibEngine(RecognitionEngine):
    name = "dlib"
    dim = 128
    threshold = 0.6

    def __init__(self, predictor, rec_model, detector):
        import dlib  # local import: dlib only needed for this engine
        self._dlib = dlib
        self._predictor = predictor
        self._rec_model = rec_model
        self._detector = detector

    def detect_and_encode(self, bgr_frame, scale=0.5):
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        faces = self._detector(small, 1)
        inv = 1.0 / scale
        out = []
        for f in faces:
            # encode on the FULL-res image for accuracy; rect scaled back
            x1, y1 = int(f.left() * inv), int(f.top() * inv)
            x2, y2 = int(f.right() * inv), int(f.bottom() * inv)
            rect_full = self._dlib.rectangle(
                max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            shape = self._predictor(rgb, rect_full)
            emb = np.array(
                self._rec_model.compute_face_descriptor(rgb, shape),
                dtype=np.float32)
            out.append(((x1, y1, x2, y2), emb))
        return out

    def match(self, embedding, known):
        if not known:
            return None
        embs = np.array([e for _, e in known], dtype=np.float32)
        dists = np.linalg.norm(embs - embedding, axis=1)
        i = int(np.argmin(dists))
        return known[i][0] if dists[i] <= self.threshold else None


class ArcFaceEngine(RecognitionEngine):
    name = "arcface"
    dim = 512
    threshold = 0.5  # cosine similarity >= 0.5 (verified: same~1.0, diff~0.25)

    def __init__(self, model_pack="buffalo_s"):
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(name=model_pack)
        self._app.prepare(ctx_id=-1)  # CPU; set ctx_id=0 for GPU

    def detect_and_encode(self, bgr_frame):
        faces = self._app.get(bgr_frame)
        out = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = np.array(f.normed_embedding, dtype=np.float32)
            out.append(((x1, y1, x2, y2), emb))
        return out

    def match(self, embedding, known):
        if not known:
            return None
        sims = [(rollno, float(np.dot(embedding, e))) for rollno, e in known]
        rollno, best = max(sims, key=lambda t: t[1])
        return rollno if best >= self.threshold else None


def get_engine():
    """Build the engine selected by RECOGNITION_ENGINE (default: dlib).

    dlib needs its models passed in (they live in app.py); arcface is
    self-contained.
    """
    choice = os.environ.get("RECOGNITION_ENGINE", "dlib").lower()
    if choice == "arcface":
        return ArcFaceEngine()
    return None  # caller builds DlibEngine with app.py's loaded models
