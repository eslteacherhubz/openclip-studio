"""Typed accessors for JSON-decoded data (mypy-strict friendly boundaries)."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def as_dict(data: object, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = data.get(key, default if default is not None else {})  # type: ignore[attr-defined]
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict at {key!r}, got {type(value).__name__}")
    return value


def as_list(data: object, key: str, default: list[Any] | None = None) -> list[Any]:
    value = data.get(key, default if default is not None else [])  # type: ignore[attr-defined]
    if not isinstance(value, list):
        raise TypeError(f"Expected list at {key!r}, got {type(value).__name__}")
    return value


def as_str(data: object, key: str, default: str = "") -> str:
    value = data.get(key, default)  # type: ignore[attr-defined]
    if not isinstance(value, str):
        raise TypeError(f"Expected str at {key!r}, got {type(value).__name__}")
    return value


def as_float(data: object, key: str, default: float = 0.0) -> float:
    value = data.get(key, default)  # type: ignore[attr-defined]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected number at {key!r}, got {type(value).__name__}")
    return float(value)


def as_int(data: object, key: str, default: int = 0) -> int:
    value = data.get(key, default)  # type: ignore[attr-defined]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected int at {key!r}, got {type(value).__name__}")
    return int(value)


def as_bool(data: object, key: str, default: bool = False) -> bool:
    value = data.get(key, default)  # type: ignore[attr-defined]
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool at {key!r}, got {type(value).__name__}")
    return value


def as_optional_str(data: object, key: str) -> str | None:
    value = data.get(key, None)  # type: ignore[attr-defined]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected str or None at {key!r}, got {type(value).__name__}")
    return value


def as_optional_list(data: object, key: str) -> list[Any] | None:
    value = data.get(key, None)  # type: ignore[attr-defined]
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"Expected list or None at {key!r}, got {type(value).__name__}")
    return value
