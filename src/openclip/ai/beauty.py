"""Beauty filter module with optional vision dependencies."""

from __future__ import annotations

try:
    import cv2
    import numpy as np

    _HAS_VISION = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HAS_VISION = False

from openclip.ai.face_analysis import FaceFrame, Rect


def beauty_available() -> bool:
    """Return True if optional vision dependencies (numpy, opencv) are installed."""
    return _HAS_VISION


def _rasterize_mask(
    rects: tuple[Rect, ...],
    h: int,
    w: int,
    feather: int = 5,
) -> np.ndarray:
    """Rasterize normalized rects into a float mask with feathered edges."""
    if not _HAS_VISION:
        raise RuntimeError("Vision dependencies (numpy, opencv) not available")

    mask = np.zeros((h, w), dtype=np.float32)
    for (nx, ny, nw, nh) in rects:
        x1 = int(nx * w)
        y1 = int(ny * h)
        x2 = int((nx + nw) * w)
        y2 = int((ny + nh) * h)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        mask[y1:y2, x1:x2] = 1.0

    if feather > 0:
        kernel_size = feather * 2 + 1
        blurred = cv2.GaussianBlur(
            mask, (kernel_size, kernel_size), sigmaX=feather / 2.0, sigmaY=feather / 2.0
        )
        mask = np.clip(blurred, 0.0, 1.0)

    return mask


def _apply_exclusions(mask: np.ndarray, zones: tuple[Rect, ...], h: int, w: int) -> np.ndarray:
    """Zero out mask inside normalized exclusion zones."""
    out = mask.copy()
    for (nx, ny, nw, nh) in zones:
        x1 = int(nx * w)
        y1 = int(ny * h)
        x2 = int((nx + nw) * w)
        y2 = int((ny + nh) * h)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        out[y1:y2, x1:x2] = 0.0
    return out


def smooth_skin(
    frame: np.ndarray,
    face_frame: FaceFrame,
    intensity: float = 0.15,
) -> np.ndarray:
    """Apply frequency-separation skin smoothing to frame.

    Uses TRUE frequency separation: low-frequency via Gaussian blur (linear low-pass),
    high-frequency = original - low, result = smoothed(low) + high. Texture is RE-ADDED.

    Args:
        frame: Input BGR image (H, W, 3) uint8.
        face_frame: FaceFrame with skin_mask_cells and exclusion_zones (normalized 0-1).
        intensity: Smoothing strength in [0, 1]. 0 = passthrough, 1 = max smoothing.

    Returns:
        Smoothed frame with identical dtype/shape to input.

    Raises:
        RuntimeError: If vision dependencies are not available.
    """
    if not _HAS_VISION:
        raise RuntimeError("Vision dependencies (numpy, opencv) not available")

    if frame.size == 0:
        return frame

    intensity = max(0.0, min(1.0, intensity))
    if intensity == 0.0:
        return frame.copy()

    h, w = frame.shape[:2]

    mask = _rasterize_mask(face_frame.skin_mask_cells, h, w)
    mask = _apply_exclusions(mask, face_frame.exclusion_zones, h, w)

    frame_f = frame.astype(np.float32) / 255.0

    # TRUE frequency separation using Gaussian blur (linear low-pass filter)
    # Base kernel size and sigma scale with intensity.
    # At intensity=0.15: ksize=7, sigma=2.0 for low-freq extraction;
    # ksize=15, sigma=6.0 for smoothing.
    # At intensity=1.0: ksize=21, sigma=7.0 for low-freq extraction;
    # ksize=31, sigma=15.0 for smoothing.
    base_ksize = int(5 + 16 * intensity)  # 5 to 21
    if base_ksize % 2 == 0:
        base_ksize += 1
    base_sigma = 1.0 + 6.0 * intensity  # 1.0 to 7.0

    # Extract low-frequency component (broad color/tone variations).
    low_freq = cv2.GaussianBlur(
        frame_f, (base_ksize, base_ksize), sigmaX=base_sigma, sigmaY=base_sigma
    )

    # High-frequency detail (texture, pores, fine lines) = original - low_freq
    high_freq = frame_f - low_freq

    # Smooth the low-frequency component more aggressively
    # Use larger kernel and sigma for stronger smoothing of blotches/gradients
    smooth_ksize = int(base_ksize * 1.5)
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1
    smooth_sigma = base_sigma * 2.0

    smoothed_low = cv2.GaussianBlur(
        low_freq, (smooth_ksize, smooth_ksize), sigmaX=smooth_sigma, sigmaY=smooth_sigma
    )

    # Reconstruct: smoothed low-freq + original high-freq (texture perfectly preserved)
    result_f = smoothed_low + high_freq

    # Blend with original using feathered mask and intensity
    out = frame.copy()
    for c in range(3):
        orig_channel = frame[:, :, c]
        result_channel = result_f[:, :, c] * 255.0
        weight = intensity * mask
        blended = orig_channel * (1.0 - weight) + result_channel * weight
        out[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)

    return out
