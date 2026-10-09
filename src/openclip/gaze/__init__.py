"""Gaze research package (optional ``gaze`` extra: numpy + opencv).

Warp-based eye-contact correction for talking-head lessons:

- detect: eye aperture + iris localization (OpenCV heuristics; Haar cascades
  for real footage, dark-pupil/threshold for controlled input)
- warp: displacement-field iris redirection (Reale et al. style warp;
  conceptually the same family of approach as NVIDIA Maxine "Eye Contact",
  implemented from scratch here)
- rig: deterministic synthetic eye images with ground-truth offsets, because
  this sandbox has no camera and no private footage may be used for tuning
- evaluate: iris-offset error, artifact PSNR, speed

All experiments and their numbers live in docs/EYE_CONTACT_RESEARCH.md.
Real-footage validation is explicitly deferred to the owner (checklist in
that document); nothing here claims to "work on real webcam video".
"""

from __future__ import annotations


def available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False
