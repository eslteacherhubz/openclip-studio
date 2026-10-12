"""Eye-aperture and iris localization.

Two operating modes:

1. ``locate_in_patch`` — dark-pupil/threshold localization inside a known
   eye patch (controlled input: the synthetic rig, or a crop the caller
   already knows is an eye). Used by the evaluation harness.
2. ``locate_in_frame`` — full-frame pipeline for real footage: Haar face
   detection → eye-region crops → same iris localization. Untested on real
   footage in this sandbox (no camera); the owner validates it with
   ``openclip gaze-analyze`` (see docs/EYE_CONTACT_RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EyeLocation:
    found: bool
    eye_cx: float  # aperture center in patch coords
    eye_cy: float
    iris_cx: float  # iris center in patch coords
    iris_cy: float
    eye_w: float  # aperture width estimate, px
    iris_r: float  # iris radius estimate, px


def _iris_by_dark_pupil(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Iris center via blobby-dark-region selection inside `mask` (a rect).

    Closing first fills the specular highlight hole inside the pupil; one
    gentle erosion removes thin dark bands (lashes, lid edges); aspect and
    compactness filters reject elongated corner wedges. Radius is scaled
    from the blob extent with a typical pupil:iris ratio (~1:2.4).
    """
    vals = gray[mask > 0]
    if vals.size == 0:
        return -1.0, -1.0, 0.0
    dark_thr = max(int(vals.min()) + 22, int(np.percentile(vals, 4)))
    dark: np.ndarray = ((gray <= dark_thr) & (mask > 0)).astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = np.asarray(cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_k), dtype=np.uint8)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = np.asarray(cv2.erode(dark, erode_k, iterations=1), dtype=np.uint8)
    if dark.max() == 0:
        return -1.0, -1.0, 0.0
    dark = np.asarray(cv2.dilate(dark, erode_k, iterations=2), dtype=np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    best, best_score, best_extent = 0, 0.0, 0.0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) <= 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > 2.5:  # elongated = lash line / corner wedge, not a pupil
            continue
        score = float(area) / (bw * bh) * area  # disk-ness * size
        if score > best_score:
            best, best_score = i, score
            best_extent = float(max(bw, bh))
    if best == 0:
        return -1.0, -1.0, 0.0
    cx = stats[best, cv2.CC_STAT_LEFT] + stats[best, cv2.CC_STAT_WIDTH] / 2.0
    cy = stats[best, cv2.CC_STAT_TOP] + stats[best, cv2.CC_STAT_HEIGHT] / 2.0
    # best_extent approximates the iris DIAMETER (pupil + dark iris rim
    # merged by closing); take the radius with a small margin.
    iris_r = max(6.0, best_extent * 0.55)
    return float(cx), float(cy), iris_r


def _aperture_by_brightness(gray: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Eye aperture estimate: brightest elongated region in the patch."""
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thr = max(int(np.percentile(blurred, 78)), 120)
    bright = (blurred >= thr).astype(np.uint8)
    bright = np.asarray(
        cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), dtype=np.uint8
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    best, best_area = 0, 0
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w > h * 1.6 and stats[i, cv2.CC_STAT_AREA] > best_area:  # elongated
            best, best_area = i, stats[i, cv2.CC_STAT_AREA]
    if best == 0:
        h_, w_ = gray.shape
        return w_ / 2, h_ / 2, float(w_ * 0.6), np.zeros(gray.shape, np.uint8)
    x, y, bw, bh, _ = stats[best]
    mask = (bright == best).astype(np.uint8)
    return x + bw / 2.0, y + bh / 2.0, float(bw), mask


def locate_in_patch(patch_bgr: np.ndarray) -> EyeLocation:
    """Locate aperture + iris inside a patch that contains the eye."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    eye_cx, eye_cy, eye_w, ap_mask = _aperture_by_brightness(gray)
    ys, xs = np.nonzero(ap_mask)
    if xs.size < 20:
        return EyeLocation(False, eye_cx, eye_cy, -1.0, -1.0, eye_w, 0.0)
    # Search INSIDE the aperture bbox rectangle: the pupil is a dark hole in
    # the bright sclera, so the bright mask itself must not clip it out.
    search = np.zeros(gray.shape, np.uint8)
    x0 = xs.min() + eye_w * 0.08
    x1 = xs.max() - eye_w * 0.08
    y0 = ys.min() + eye_w * 0.06
    y1 = ys.max() - eye_w * 0.10
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    search[(xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)] = 1
    iris_cx, iris_cy, iris_r = _iris_by_dark_pupil(gray, search)
    found = iris_cx >= 0 and eye_w > 10
    return EyeLocation(found, eye_cx, eye_cy, iris_cx, iris_cy, eye_w, iris_r)


_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


def locate_in_frame(frame_bgr: np.ndarray) -> list[EyeLocation]:
    """Real-footage pipeline: face → both eye patches → iris localization.

    Returns up to two EyeLocations with iris coordinates in FRAME coords.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade().detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
    out: list[EyeLocation] = []
    for fx, fy, fw, fh in faces[:1]:  # one face (talking head)
        for ecx, ecy, ew, eh in (
            (fx + int(0.14 * fw), fy + int(0.28 * fh), int(0.24 * fw), int(0.16 * fh)),
            (fx + int(0.62 * fw), fy + int(0.28 * fh), int(0.24 * fw), int(0.16 * fh)),
        ):
            patch = frame_bgr[ecy : ecy + eh, ecx : ecx + ew]
            if patch.size == 0:
                continue
            loc = locate_in_patch(patch)
            if loc.found:
                out.append(
                    EyeLocation(
                        found=True,
                        eye_cx=ecx + loc.eye_cx,
                        eye_cy=ecy + loc.eye_cy,
                        iris_cx=ecx + loc.iris_cx,
                        iris_cy=ecy + loc.iris_cy,
                        eye_w=loc.eye_w,
                        iris_r=loc.iris_r,
                    )
                )
    return out
