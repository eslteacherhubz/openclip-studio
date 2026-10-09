"""Audio enhancement presets built from FFmpeg filters (all local)."""

from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "none": [],
    "teaching": [
        "highpass=f=70",
        "afftdn=nr=12:nf=-25",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ],
    "denoise": [
        "highpass=f=70",
        "afftdn=nr=18:nf=-30",
    ],
    "normalize": [
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ],
}

PRESET_NAMES = tuple(PRESETS.keys())


def build_audio_filter(preset: str) -> str | None:
    """Return the -af chain for a preset, or None if passthrough."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown audio preset {preset!r}; known: {PRESET_NAMES}")
    chain = PRESETS[preset]
    return ",".join(chain) if chain else None
