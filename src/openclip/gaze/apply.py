"""Video-level gaze correction: analyze and apply.

Decoding via OpenCV's bundled FFmpeg; encoding via an FFmpeg subprocess
pipe (H.264 + original audio remuxed). ``gaze-analyze`` reports what the
detector sees (for the owner's real-footage validation); ``gaze-apply``
produces a corrected video. Both are research tools behind the ``gaze``
extra — the default pipeline never enables them silently.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from openclip.gaze.detect import EyeLocation, locate_in_frame, locate_in_patch
from openclip.gaze.warp import TemporalSmoother, warp_frame


@dataclass
class GazeStats:
    frames: int
    frames_with_eyes: int
    mean_offset_pct: float  # mean |iris-eye offset| as % of eye width
    max_offset_pct: float


def _ffmpeg_encode_cmd(out_path: Path, w: int, h: int, fps: float) -> list[str]:
    from openclip.engine.ffmpeg import ffmpeg_exe

    return [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]


def analyze_video(
    media: str | Path,
    max_frames: int | None = None,
    mode: str = "frame",
) -> GazeStats:
    """Measure detected gaze offsets without modifying anything."""
    cap = cv2.VideoCapture(str(media))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {media}")
    smoother = TemporalSmoother()
    frames = 0
    with_eyes = 0
    offsets: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and frames >= max_frames):
            break
        frames += 1
        locs = (
            locate_in_frame(frame) if mode == "frame" else _patch_locs(frame)
        )
        locs = smoother.smooth(locs)
        if locs:
            with_eyes += 1
            offsets.extend(_offsets_of(locs))
    cap.release()
    return GazeStats(
        frames=frames,
        frames_with_eyes=with_eyes,
        mean_offset_pct=float(np.mean(offsets)) if offsets else 0.0,
        max_offset_pct=float(np.max(offsets)) if offsets else 0.0,
    )


def _patch_locs(frame: np.ndarray) -> list[EyeLocation]:
    """Treat the whole frame as one eye patch (rig/evaluation mode)."""
    loc = locate_in_patch(frame)
    return [loc] if loc.found else []


def _offsets_of(locs: list[EyeLocation]) -> list[float]:
    out: list[float] = []
    for loc in locs:
        out.append(
            float(
                np.hypot(loc.iris_cx - loc.eye_cx, loc.iris_cy - loc.eye_cy)
                / max(loc.eye_w, 1.0)
                * 100.0
            )
        )
    return out


def apply_to_video(
    media: str | Path,
    out_path: str | Path,
    strength: float = 0.8,
    max_frames: int | None = None,
    mode: str = "frame",
) -> GazeStats:
    """Write a gaze-corrected copy of `media` to `out_path` (H.264 MP4)."""
    media_p = Path(media)
    cap = cv2.VideoCapture(str(media_p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {media_p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    encode = subprocess.Popen(
        _ffmpeg_encode_cmd(out_p, w, h, fps), stdin=subprocess.PIPE
    )
    smoother = TemporalSmoother()
    frames = 0
    with_eyes = 0
    offsets: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and frames >= max_frames):
                break
            frames += 1
            locs = (
                locate_in_frame(frame) if mode == "frame" else _patch_locs(frame)
            )
            locs = smoother.smooth(locs)
            if locs:
                with_eyes += 1
                offsets.extend(_offsets_of(locs))
                frame = warp_frame(frame, locs, strength)
            assert encode.stdin is not None
            encode.stdin.write(frame.astype(np.uint8).tobytes())
    finally:
        cap.release()
        if encode.stdin is not None:
            encode.stdin.close()
        encode.wait(timeout=300)
    _remux_audio(media_p, out_p)
    return GazeStats(
        frames=frames,
        frames_with_eyes=with_eyes,
        mean_offset_pct=float(np.mean(offsets)) if offsets else 0.0,
        max_offset_pct=float(np.max(offsets)) if offsets else 0.0,
    )


def _remux_audio(source: Path, out_path: Path) -> None:
    """Copy the original audio track onto the corrected video, if any."""
    from openclip.engine.ffmpeg import probe, run_ffmpeg

    if not probe(source).has_audio:
        return
    tmp = out_path.with_suffix(".tmp.mp4")
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(out_path),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c",
            "copy",
            "-shortest",
            str(tmp),
        ]
    )
    tmp.replace(out_path)


def analyze_to_json(media: str | Path, max_frames: int | None = None) -> str:
    return json.dumps(asdict(analyze_video(media, max_frames=max_frames)), indent=2)
