"""Error handling, input validation, and user-friendly error messages for OpenClip.

This module provides a small exception hierarchy and validation utilities that
convert raw Python exceptions into clear, actionable user messages while
preserving full tracebacks in logs for debugging.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Exception Hierarchy
# ============================================================================


class OpenClipError(Exception):
    """Base exception for all user-facing OpenClip errors.

    Carries a user-friendly message and an optional exit code.
    The full traceback is logged automatically when caught by the error handler.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code

    def __str__(self) -> str:
        return self.message


class InputError(OpenClipError):
    """Raised when user provides invalid input (bad path, malformed JSON, etc.)."""
    pass


class DependencyError(OpenClipError):
    """Raised when an optional dependency (vision/gui/asr/gaze) is missing."""
    pass


class MediaError(OpenClipError):
    """Raised when media file is invalid, missing, or unsuitable."""
    pass


class RenderError(OpenClipError):
    """Raised when rendering fails for a known reason (ffmpeg, codec, etc.)."""
    pass


class CancellationError(OpenClipError):
    """Raised when the user cancels a long-running operation."""
    pass


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_media(path: str | Path) -> None:
    """Validate that a media file exists, is readable, and has a video stream.

    Args:
        path: Path to the media file.

    Raises:
        MediaError: If the file doesn't exist, isn't a video, has no video stream,
            or is too short (< 0.5s).
    """
    from openclip.engine.ffmpeg import FFmpegError, probe

    path = Path(path)

    if not path.exists():
        raise MediaError(f"Video file not found: {path}")

    if not path.is_file():
        raise MediaError(f"Path is not a file: {path}")

    # Check readability
    if not os.access(path, os.R_OK):
        raise MediaError(f"Video file not readable (permission denied): {path}")

    try:
        info = probe(str(path))
    except FFmpegError as exc:
        raise MediaError(f"Failed to probe video file {path}: {exc}") from exc
    except Exception as exc:
        logger.debug("Probe failed for %s", path, exc_info=True)
        raise MediaError(f"Cannot read video file {path}: {exc}") from exc

    if not info.has_video:
        raise MediaError(f"File has no video stream: {path}")

    if info.duration_s <= 0.5:
        raise MediaError(
            f"Video too short ({info.duration_s:.2f}s): minimum duration is 0.5s"
        )

    if info.width is None or info.height is None or info.width <= 0 or info.height <= 0:
        raise MediaError(f"Invalid video dimensions: {info.width}x{info.height}")

    logger.debug(
        "Media validation passed: %s (%.2fs, %dx%d)",
        path,
        info.duration_s,
        info.width,
        info.height,
    )


def validate_dependencies(needed: set[str]) -> None:
    """Check that optional dependencies are available.

    Args:
        needed: Set of extra names (e.g., {"vision", "gui", "asr", "gaze"}).

    Raises:
        DependencyError: If any required extra is missing or unknown, with the pip install command.
    """
    known_extras = {"vision", "gui", "asr", "gaze"}
    unknown = needed - known_extras
    if unknown:
        raise DependencyError(
            "Unknown dependency extras: "
            f"{', '.join(sorted(unknown))}. "
            f"Known extras: {', '.join(sorted(known_extras))}"
        )

    missing = []
    for extra in sorted(needed):
        if extra == "vision" and not (
            importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy")
        ):
            missing.append("vision (pip install eterna-openclip-studio[vision])")
        elif extra == "gui" and not importlib.util.find_spec("PySide6"):
            missing.append("gui (pip install eterna-openclip-studio[gui])")
        elif extra == "asr" and not importlib.util.find_spec("faster_whisper"):
            missing.append("asr (pip install eterna-openclip-studio[asr])")
        elif extra == "gaze" and not importlib.util.find_spec("mediapipe"):
            missing.append("gaze (pip install eterna-openclip-studio[gaze])")

    if missing:
        raise DependencyError(
            "Missing optional dependencies:\n  " + "\n  ".join(missing)
        )


def get_missing_dependencies() -> dict[str, str]:
    """Check all optional dependencies and return a dict of extra -> install command."""
    result = {}
    checks = {
        "vision": (["cv2", "numpy"], "pip install eterna-openclip-studio[vision]"),
        "gui": (["PySide6"], "pip install eterna-openclip-studio[gui]"),
        "asr": (["faster_whisper"], "pip install eterna-openclip-studio[asr]"),
        "gaze": (["mediapipe"], "pip install eterna-openclip-studio[gaze]"),
    }
    for extra, (mods, install_cmd) in checks.items():
        if all(importlib.util.find_spec(mod) for mod in mods):
            result[extra] = "✅ Available"
        else:
            result[extra] = f"❌ Missing — {install_cmd}"
    return result


# ============================================================================
# Error Handling Wrapper
# ============================================================================


def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that catches OpenClipError and logs traceback, prints friendly message.

    For unexpected exceptions, logs full traceback and prints a generic message
    directing the user to the log file.
    """
    def wrapper(*args: object, **kwargs: object) -> T:
        try:
            return func(*args, **kwargs)
        except OpenClipError as exc:
            # Friendly error: log traceback, print user message
            logger.error("%s: %s", exc.__class__.__name__, exc.message, exc_info=True)
            print(f"Error: {exc.message}", file=sys.stderr)
            sys.exit(exc.exit_code)
        except KeyboardInterrupt:
            print("\nCancelled by user", file=sys.stderr)
            sys.exit(130)
        except Exception as exc:
            # Unexpected error: log full traceback, print generic message
            log_path = _write_crash_log(exc)
            logger.exception("Unexpected error in %s", func.__name__)
            print(
                f"Unexpected error: {exc}\n"
                f"Full traceback written to: {log_path}\n"
                f"Please report this issue at https://github.com/anomalyco/eterna-openclip-studio/issues",
                file=sys.stderr,
            )
            sys.exit(1)
    return wrapper


def _write_crash_log(exc: Exception) -> Path:
    """Write crash log to a file in the temp directory."""
    import tempfile
    from datetime import datetime

    log_dir = Path(tempfile.gettempdir()) / "openclip-crash-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"crash-{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("OpenClip Crash Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write(f"{'=' * 50}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    return log_path


def run_with_error_handling(
    func: Callable[[], T],
    log_path: Path | None = None,
) -> T:
    """Run a function with full error handling, optionally writing crash log to a specific path.

    This is a lower-level API for use when you don't want the decorator approach.
    """
    try:
        return func()
    except OpenClipError as exc:
        logger.error("%s: %s", exc.__class__.__name__, exc.message, exc_info=True)
        print(f"Error: {exc.message}", file=sys.stderr)
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        print("\nCancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if log_path:
            _write_crash_log_to_path(exc, log_path)
        else:
            log_path = _write_crash_log(exc)
        logger.exception("Unexpected error")
        print(
            f"Unexpected error: {exc}\n"
            f"Full traceback written to: {log_path}\n"
            f"Please report this issue at https://github.com/anomalyco/eterna-openclip-studio/issues",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_crash_log_to_path(exc: Exception, log_path: Path) -> None:
    """Write crash log to a specific path."""
    from datetime import datetime

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("OpenClip Crash Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write(f"{'=' * 50}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


# ============================================================================
# Convenience Re-exports
# ============================================================================

__all__ = [
    "CancellationError",
    "DependencyError",
    "InputError",
    "MediaError",
    "OpenClipError",
    "RenderError",
    "get_missing_dependencies",
    "handle_errors",
    "run_with_error_handling",
    "validate_dependencies",
    "validate_media",
]
