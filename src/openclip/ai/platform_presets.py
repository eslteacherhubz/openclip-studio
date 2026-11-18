"""Platform export presets: one-click export to TikTok, Reels, Shorts, YouTube, etc.

Presets define target resolution, aspect ratio, duration limits, and loudness.
Vertical platforms (9:16) automatically enable smart reframe so the subject
stays centered. Registry is pure data — no hardcoded if/else branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AspectRatio = Literal["9:16", "16:9", "1:1", "4:5"]


@dataclass(frozen=True)
class PlatformPreset:
    """Export preset for a specific platform."""

    name: str
    target_width: int
    target_height: int
    aspect: AspectRatio
    max_duration_s: float | None  # None = no limit
    vertical: bool  # True for 9:16 (enables smart reframe)
    loudness_lufs: float = -14.0  # Target integrated loudness (EBU R128)
    notes: str = ""

    @property
    def duration_warning_threshold_s(self) -> float | None:
        """Threshold at which to warn about source exceeding platform limit."""
        if self.max_duration_s is None:
            return None
        # Warn at 80% of max duration
        return self.max_duration_s * 0.8


#: Registry of known platform presets
PRESETS: dict[str, PlatformPreset] = {
    "tiktok": PlatformPreset(
        name="tiktok",
        target_width=1080,
        target_height=1920,
        aspect="9:16",
        max_duration_s=600.0,  # 10 minutes
        vertical=True,
        loudness_lufs=-14.0,
        notes="TikTok supports up to 10 minutes; vertical video recommended",
    ),
    "reels": PlatformPreset(
        name="reels",
        target_width=1080,
        target_height=1920,
        aspect="9:16",
        max_duration_s=90.0,
        vertical=True,
        loudness_lufs=-14.0,
        notes="Instagram Reels max 90 seconds",
    ),
    "shorts": PlatformPreset(
        name="shorts",
        target_width=1080,
        target_height=1920,
        aspect="9:16",
        max_duration_s=60.0,
        vertical=True,
        loudness_lufs=-14.0,
        notes="YouTube Shorts max 60 seconds",
    ),
    "youtube": PlatformPreset(
        name="youtube",
        target_width=1920,
        target_height=1080,
        aspect="16:9",
        max_duration_s=None,
        vertical=False,
        loudness_lufs=-14.0,
        notes="Standard YouTube upload (16:9, no duration limit)",
    ),
    "instagram": PlatformPreset(
        name="instagram",
        target_width=1080,
        target_height=1350,
        aspect="4:5",
        max_duration_s=60.0,  # Feed video max 60s
        vertical=False,
        loudness_lufs=-14.0,
        notes="Instagram Feed (4:5 portrait); Reels uses 9:16",
    ),
    "linkedin": PlatformPreset(
        name="linkedin",
        target_width=1920,
        target_height=1080,
        aspect="16:9",
        max_duration_s=600.0,  # 10 minutes
        vertical=False,
        loudness_lufs=-14.0,
        notes="LinkedIn video posts (16:9 recommended)",
    ),
    "twitter": PlatformPreset(
        name="twitter",
        target_width=1280,
        target_height=720,
        aspect="16:9",
        max_duration_s=140.0,  # 2 min 20s
        vertical=False,
        loudness_lufs=-14.0,
        notes="X/Twitter video (16:9, max 2:20)",
    ),
}


def get_preset(name: str) -> PlatformPreset:
    """Get a preset by name (case-insensitive).

    Args:
        name: Platform name (e.g., "tiktok", "REELS", "Shorts")

    Returns:
        The PlatformPreset for the platform.

    Raises:
        ValueError: If the platform name is unknown.
    """
    key = name.lower().strip()
    if key not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown platform preset: {name!r}. Valid options: {valid}"
        )
    return PRESETS[key]


def list_presets() -> list[PlatformPreset]:
    """Return all presets sorted by name."""
    return sorted(PRESETS.values(), key=lambda p: p.name)


def preset_names() -> list[str]:
    """Return all preset names sorted alphabetically."""
    return sorted(PRESETS.keys())


__all__ = [
    "PRESETS",
    "AspectRatio",
    "PlatformPreset",
    "get_preset",
    "list_presets",
    "preset_names",
]
