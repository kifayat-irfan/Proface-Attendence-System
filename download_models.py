"""Download the dlib shape predictor model required by ProFace.

The 68-point facial landmark model (~95MB) is too large to ship in the
repository, so this script fetches it from the official dlib mirror and
verifies it. Run once before starting the app:

    python download_models.py
"""
import bz2
import os
import shutil
import sys
import urllib.request

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"
EXPECTED_SIZE = 99_557_280  # ~95MB uncompressed


def _progress(blocks, block_size, total):
    done = blocks * block_size
    pct = min(100, done * 100 / total) if total > 0 else 0
    sys.stdout.write(f"\rDownloading {MODEL_FILE}: {pct:5.1f}%")
    sys.stdout.flush()


def main():
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
    if os.path.isfile(target) and os.path.getsize(target) == EXPECTED_SIZE:
        print(f"{MODEL_FILE} already present, skipping download.")
        return

    archive = target + ".bz2"
    print(f"Downloading {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, archive, _progress)
    print("\nExtracting...")
    with bz2.open(archive, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(archive)

    size = os.path.getsize(target)
    if size != EXPECTED_SIZE:
        print(f"WARNING: unexpected file size ({size} bytes). The download may be corrupt.")
    else:
        print(f"Done: {MODEL_FILE} ({size // (1024 * 1024)}MB)")


if __name__ == "__main__":
    main()
