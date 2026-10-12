"""FFmpeg/ffprobe process wrappers used by every media operation.

FFmpeg is invoked strictly as a subprocess (never linked), so FFmpeg's own
GPL/LGPL build configuration never restricts this codebase (GPL-3+).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class FFmpegError(RuntimeError):
    """A wrapped, diagnostics-friendly ffmpeg/ffprobe failure."""


def ffmpeg_exe() -> str:
    """Locate the ffmpeg binary: env override, PATH, then imageio-ffmpeg wheel."""
    env = os.environ.get("OPENCLIP_FFMPEG")
    if env:
        return env
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg  # base dependency; ships a static ffmpeg binary

        exe: str = imageio_ffmpeg.get_ffmpeg_exe()
        return exe
    except Exception as exc:  # pragma: no cover - environment-specific
        raise FFmpegError(
            "No ffmpeg found. Install FFmpeg, or set OPENCLIP_FFMPEG to its path."
        ) from exc


def ffprobe_exe() -> str | None:
    """Locate ffprobe (env override, then PATH). imageio-ffmpeg ships no ffprobe."""
    env = os.environ.get("OPENCLIP_FFPROBE")
    if env:
        return env
    return shutil.which("ffprobe")


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"Command timed out: {' '.join(cmd[:6])}...") from exc
    except OSError as exc:
        raise FFmpegError(f"Failed to run {cmd[0]}: {exc}") from exc
    return proc


def run_ffmpeg(args: list[str], timeout: float | None = None) -> None:
    """Run ffmpeg with `-hide_banner -nostdin` prepended; raise on failure."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", *args]
    proc = _run(cmd, timeout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise FFmpegError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


def run_ffmpeg_capture(
    args: list[str], timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg and return the completed process without checking the exit code.

    Used for informational invocations (e.g. `ffmpeg -i`, `silencedetect`)
    where diagnostics live in stderr and a nonzero exit may be expected.
    """
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", *args]
    return _run(cmd, timeout)


@dataclass
class MediaInfo:
    """Everything the engine needs to know about a media file."""

    path: Path
    duration_s: float
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    pix_fmt: str = ""
    acodec: str = ""
    sample_rate: int = 0
    channels: int = 0
    probe_source: str = "ffmpeg"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "duration_s": self.duration_s,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vcodec": self.vcodec,
            "pix_fmt": self.pix_fmt,
            "acodec": self.acodec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "probe_source": self.probe_source,
        }


def probe(path: str | Path) -> MediaInfo:
    """Probe a media file. Prefers ffprobe JSON; falls back to parsing `ffmpeg -i`."""
    p = Path(path)
    if not p.exists():
        raise FFmpegError(f"No such media file: {p}")
    exe = ffprobe_exe()
    if exe:
        info = _probe_ffprobe(exe, p)
        if info is not None:
            return info
    return _probe_ffmpeg(p)


def _probe_ffprobe(exe: str, p: Path) -> MediaInfo | None:
    cmd = [
        exe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    proc = _run(cmd, timeout=60)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    duration = 0.0
    fmt = data.get("format", {})
    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    info = MediaInfo(path=p, duration_s=duration, probe_source="ffprobe")
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not info.has_video:
            info.has_video = True
            info.width = int(stream.get("width", 0) or 0)
            info.height = int(stream.get("height", 0) or 0)
            info.vcodec = str(stream.get("codec_name", ""))
            info.pix_fmt = str(stream.get("pix_fmt", ""))
            rate = str(stream.get("avg_frame_rate", "0/1"))
            try:
                num, _, den = rate.partition("/")
                fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            info.fps = fps
        elif stream.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.acodec = str(stream.get("codec_name", ""))
            try:
                info.sample_rate = int(stream.get("sample_rate", 0) or 0)
            except (TypeError, ValueError):
                info.sample_rate = 0
            info.channels = int(stream.get("channels", 0) or 0)
    if info.duration_s <= 0.0 and info.has_video:
        # Some demuxers report no container duration; trust stream duration.
        for stream in data.get("streams", []):
            try:
                d = float(stream.get("duration", 0) or 0)
            except (TypeError, ValueError):
                d = 0.0
            info.duration_s = max(info.duration_s, d)
    return info


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(
    r"Stream #\d+:\d+.*: Video: (\w+).*?, (\w+)(?:\(\w+\))?, (\d{2,5})x(\d{2,5})"
)
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUD_RE = re.compile(
    r"Stream #\d+:\d+.*: Audio: (\w+).*?, (\d+) Hz.*?, (?:mono|stereo|\d+ channels)"
)
_AUD_CHANNELS_RE = re.compile(r", (mono|stereo|\d+ channels)")


def _probe_ffmpeg(p: Path) -> MediaInfo:
    """Fallback probe by parsing `ffmpeg -i` banner output (no ffprobe needed)."""
    proc = run_ffmpeg_capture(["-i", str(p)], timeout=60)
    err = proc.stderr
    info = MediaInfo(path=p, duration_s=0.0, probe_source="ffmpeg")
    if m := _DUR_RE.search(err):
        h, mi, s = m.groups()
        info.duration_s = int(h) * 3600 + int(mi) * 60 + float(s)
    if m := _VID_RE.search(err):
        info.has_video = True
        info.vcodec = m.group(1)
        info.pix_fmt = m.group(2)
        info.width, info.height = int(m.group(3)), int(m.group(4))
        if fm := _FPS_RE.search(err[m.end() :]):
            info.fps = float(fm.group(1))
    if m := _AUD_RE.search(err):
        info.has_audio = True
        info.acodec = m.group(1)
        try:
            info.sample_rate = int(m.group(2))
        except ValueError:
            info.sample_rate = 0
        cm = _AUD_CHANNELS_RE.search(err[m.start() :])
        if cm:
            tok = cm.group(1)
            info.channels = 1 if tok == "mono" else 2 if tok == "stereo" else int(tok.split()[0])
    if info.duration_s <= 0:
        raise FFmpegError(f"Could not determine duration of {p}")
    return info


def escape_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument.

    Handles the colon-in-Windows-drive and quotes pitfalls. Returns the path
    as it should appear inside the filter string (caller adds `filename=`).
    """
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\:").replace("'", "\\'")
    return s


@dataclass
class SilenceSpan:
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class SilenceResult:
    spans: list[SilenceSpan] = field(default_factory=list)


def detect_silences(
    media: str | Path,
    noise_db: float = -35.0,
    min_duration_s: float = 0.5,
) -> list[SilenceSpan]:
    """Detect silent spans with ffmpeg's `silencedetect` audio filter."""
    proc = run_ffmpeg_capture(
        [
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration_s}",
            "-f",
            "null",
            "-",
        ],
        timeout=600,
    )
    err = proc.stderr
    spans: list[SilenceSpan] = []
    start: float | None = None
    for line in err.splitlines():
        line = line.strip()
        if m := re.search(r"silence_start:\s*([\d.]+)", line):
            start = float(m.group(1))
        elif m := re.search(r"silence_end:\s*([\d.]+)", line):
            end = float(m.group(1))
            if start is not None:
                spans.append(SilenceSpan(start_s=start, end_s=end))
                start = None
    if start is not None:
        # Media ends in silence: ffmpeg reports no silence_end for the tail.
        spans.append(SilenceSpan(start_s=start, end_s=start + min_duration_s))
    return spans
