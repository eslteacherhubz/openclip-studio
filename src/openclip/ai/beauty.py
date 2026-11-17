"""Beauty filter module with optional vision dependencies."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

try:
    import cv2
    import numpy as np

    _HAS_VISION = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HAS_VISION = False

from openclip.ai.face_analysis import CachedFaceAnalysis, FaceFrame, Rect


@dataclass(frozen=True)
class BeautyConfig:
    """Beauty/retouching pre-pass configuration; all intensities clamped to [0, 1]."""

    skin_smooth: float = 0.15
    jaw_slim: float = 0.10
    neck_tighten: float = 0.10
    blemishes: tuple[Rect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "skin_smooth", max(0.0, min(1.0, float(self.skin_smooth)))
        )
        object.__setattr__(
            self, "jaw_slim", max(0.0, min(1.0, float(self.jaw_slim)))
        )
        object.__setattr__(
            self, "neck_tighten", max(0.0, min(1.0, float(self.neck_tighten)))
        )


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


def micro_sculpt(
    frame: np.ndarray,
    face_frame: FaceFrame,
    jaw_slim: float = 0.10,
    neck_tighten: float = 0.10,
) -> np.ndarray:
    """Apply gentle jaw/neck tightening via sparse landmark-driven warping.

    Uses cv2.remap with a smooth displacement field derived from
    ``face_frame.jaw_neck_landmarks``. The warp is confined to the face/neck
    region with a radial falloff so background pixels remain byte-identical.

    Args:
        frame: Input BGR image (H, W, 3) uint8.
        face_frame: FaceFrame with jaw_neck_landmarks (normalized 0-1).
        jaw_slim: Jaw slimming strength in [0, 1]. 0 = passthrough.
        neck_tighten: Neck tightening strength in [0, 1]. 0 = passthrough.

    Returns:
        Warped frame with identical dtype/shape to input.

    Raises:
        RuntimeError: If vision dependencies are not available.
    """
    if not _HAS_VISION:
        raise RuntimeError("Vision dependencies (numpy, opencv) not available")

    if frame.size == 0:
        return frame

    jaw_slim = max(0.0, min(1.0, jaw_slim))
    neck_tighten = max(0.0, min(1.0, neck_tighten))
    if jaw_slim == 0.0 and neck_tighten == 0.0:
        return frame.copy()

    h, w = frame.shape[:2]
    max_disp = 0.015 * w  # 1.5% of frame width

    # Build influence mask from skin_mask_cells (same region as smooth_skin)
    mask = _rasterize_mask(face_frame.skin_mask_cells, h, w)
    mask = _apply_exclusions(mask, face_frame.exclusion_zones, h, w)

    # Extend mask downward to cover neck region
    # Find bottom of skin mask and extend
    mask_indices = np.where(mask > 0)
    if mask_indices[0].size > 0:
        bottom_y = int(mask_indices[0].max())
        neck_extension = int(0.15 * h)  # extend ~15% of frame height down
        neck_bottom = min(h, bottom_y + neck_extension)
        mask[bottom_y:neck_bottom, :] = np.maximum(
            mask[bottom_y:neck_bottom, :],
            np.linspace(1.0, 0.0, neck_bottom - bottom_y)[:, np.newaxis],
        )

    # Feather the mask further for smooth falloff
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.02, sigmaY=h * 0.02)
    mask = np.clip(mask, 0.0, 1.0)

    # Initialize displacement maps
    base_x = np.arange(w, dtype=np.float32)
    base_y = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(base_x, base_y)

    disp_x = np.zeros((h, w), dtype=np.float32)
    disp_y = np.zeros((h, w), dtype=np.float32)

    # Process jaw landmarks (first 6 points) for jaw slimming
    landmarks = face_frame.jaw_neck_landmarks
    if jaw_slim > 0.0:
        for _i, pt in enumerate(landmarks[:6]):
            if pt is None:
                continue
            lx, ly = pt
            px = int(lx * w)
            py = int(ly * h)
            # Displace horizontally toward center (slimming)
            dx = (face_frame.center_x * w - px) * jaw_slim
            dx = np.clip(dx, -max_disp, max_disp)
            # Apply with radial falloff
            dy_grid, dx_grid = np.ogrid[:h, :w]
            dist = np.sqrt((dx_grid - px) ** 2 + (dy_grid - py) ** 2)
            falloff = np.exp(-(dist / (w * 0.12)) ** 2)
            disp_x += dx * falloff * mask

    # Process neck landmarks (last 2 points) for neck tightening
    if neck_tighten > 0.0:
        for _i, pt in enumerate(landmarks[6:8]):
            if pt is None:
                continue
            lx, ly = pt
            px = int(lx * w)
            py = int(ly * h)
            # Displace horizontally toward center (tightening)
            dx = (face_frame.center_x * w - px) * neck_tighten
            dx = np.clip(dx, -max_disp, max_disp)
            # Apply with radial falloff
            dy_grid, dx_grid = np.ogrid[:h, :w]
            dist = np.sqrt((dx_grid - px) ** 2 + (dy_grid - py) ** 2)
            falloff = np.exp(-(dist / (w * 0.15)) ** 2)
            disp_x += dx * falloff * mask

    # Apply displacement
    map_x = map_x + disp_x
    map_y = map_y + disp_y

    # Clamp to valid range
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # Remap using cv2.remap (INTER_LINEAR for quality)
    out = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return out


def spot_heal(
    frame: np.ndarray,
    face_frame: FaceFrame,
    blemishes: Sequence[Rect] = (),
) -> np.ndarray:
    """Inpaint blemishes that track the face center.

    ``blemishes`` are normalized rects ``(rel_x, rel_y, w, h)`` where
    ``rel_x``/``rel_y`` are offsets from ``face_frame.center_x``/``center_y``,
    so the box follows the face as it moves. ``w``/``h`` are normalized to
    frame width/height.

    Args:
        frame: Input BGR image (H, W, 3) uint8.
        face_frame: FaceFrame providing the anchor center.
        blemishes: Sequence of relative normalized blemish rects. Empty = passthrough.

    Returns:
        Inpainted frame with identical dtype/shape to input.

    Raises:
        RuntimeError: If vision dependencies are not available.
    """
    if not _HAS_VISION:
        raise RuntimeError("Vision dependencies (numpy, opencv) not available")

    if frame.size == 0:
        return frame

    if len(blemishes) == 0:
        return frame.copy()

    h, w = frame.shape[:2]

    # Convert relative normalized rects to absolute pixel coordinates
    mask = np.zeros((h, w), dtype=np.uint8)
    for (rel_x, rel_y, bw, bh) in blemishes:
        abs_nx = face_frame.center_x + rel_x
        abs_ny = face_frame.center_y + rel_y
        x1 = int(abs_nx * w)
        y1 = int(abs_ny * h)
        x2 = int((abs_nx + bw) * w)
        y2 = int((abs_ny + bh) * h)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        mask[y1:y2, x1:x2] = 255

    if not mask.any():
        return frame.copy()

    # Inpaint with TELEA algorithm
    inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Feather mask edges for seamless blend
    feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
    feathered = feathered.astype(np.float32) / 255.0
    feathered = np.clip(feathered, 0.0, 1.0)

    # Blend inpainted result with original through feathered mask
    out = frame.copy()
    for c in range(3):
        orig_ch = frame[:, :, c].astype(np.float32)
        inpaint_ch = inpainted[:, :, c].astype(np.float32)
        blended = orig_ch * (1.0 - feathered) + inpaint_ch * feathered
        out[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def beauty_prepass(
    source_path: str,
    output_path: str,
    config: BeautyConfig,
    face_analysis: CachedFaceAnalysis,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Decode ``source_path``, apply the beauty chain per-frame, re-encode to ``output_path``.

    For each frame the chain ``smooth_skin -> micro_sculpt -> spot_heal`` is
    applied using the face analysis for that frame. Frames with no detected
    face are written through unchanged. Decoded frames are piped as raw BGR
    into an FFmpeg encoder (libx264, CRF 16) so the intermediate is true
    high-quality H.264 rather than a lossy cv2 codec fallback; the source
    audio stream is muxed in unchanged.

    Raises:
        RuntimeError: If vision dependencies (numpy, opencv) are unavailable,
            the source cannot be opened, or the FFmpeg encoder fails.
    """
    if not beauty_available():
        raise RuntimeError("Vision dependencies (numpy, opencv) not available")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source_path}")

    ffmpeg_args = [
        "ffmpeg", "-y",
        "-nostats",
        "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", source_path,
        "-map", "0:v",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    proc = subprocess.Popen(
        ffmpeg_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        shell=False,
    )
    assert proc.stdin is not None and proc.stderr is not None

    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_index / fps if fps > 0 else 0.0
            face_frame = face_analysis.analyze_frame(
                frame_index, timestamp, frame
            )
            if face_frame is None:
                out = frame
            else:
                out = smooth_skin(frame, face_frame, intensity=config.skin_smooth)
                out = micro_sculpt(
                    out,
                    face_frame,
                    jaw_slim=config.jaw_slim,
                    neck_tighten=config.neck_tighten,
                )
                out = spot_heal(out, face_frame, blemishes=config.blemishes)
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            frame_index += 1
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        proc.stdin.close()
        returncode = proc.wait()

    if returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg encode failed (rc={returncode}): {stderr or 'no stderr'}"
        )
