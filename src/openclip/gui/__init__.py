"""GUI package (optional ``gui`` extra). Never imported by the engine."""

from __future__ import annotations


def available() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False
