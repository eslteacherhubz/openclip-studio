"""Temporal smoothing primitives (EWMA and moving average).

Raw per-frame detection output jitters frame to frame; that jitter becomes
visible flicker in reframing crops and would destroy beauty-filter quality.
Every stream the face-analysis layer emits — crop anchors, landmark
coordinates, exclusion-zone rectangles, mask cells — is smoothed with the
primitives in this module before any filter is generated.

Pure stdlib, deterministic, no dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

#: Default EWMA weight of the newest sample. 1.0 disables smoothing;
#: smaller values smooth harder (at the cost of a little lag).
DEFAULT_SMOOTH_ALPHA: float = 0.3


def validate_alpha(alpha: float) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha


@dataclass
class EWMA:
    """Streaming exponentially-weighted moving average.

    ``alpha`` is the weight of the newest sample: ``update`` returns
    ``alpha*x + (1-alpha)*prev``. The first sample initializes the state.
    """

    alpha: float = DEFAULT_SMOOTH_ALPHA
    value: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha)

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def smooth_series(values: Sequence[float], alpha: float = DEFAULT_SMOOTH_ALPHA) -> list[float]:
    """Batch EWMA over a series; output has the same length as the input."""
    validate_alpha(alpha)
    state = EWMA(alpha)
    return [state.update(v) for v in values]


def moving_average_series(
    values: Sequence[float], window: int = 3
) -> list[float]:
    """Trailing moving average with a fixed ``window`` (causal, same length)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo : i + 1]
        out.append(sum(seg) / len(seg))
    return out


def smooth_pairs(
    points: Sequence[tuple[float, float]], alpha: float = DEFAULT_SMOOTH_ALPHA
) -> list[tuple[float, float]]:
    """EWMA-smooth a series of 2D points coordinate-wise."""
    xs = smooth_series([p[0] for p in points], alpha)
    ys = smooth_series([p[1] for p in points], alpha)
    return list(zip(xs, ys, strict=True))


def smooth_optional_pairs(
    points: Sequence[tuple[float, float] | None], alpha: float = DEFAULT_SMOOTH_ALPHA
) -> list[tuple[float, float] | None]:
    """EWMA over a 2D series that may contain ``None`` (detection dropouts).

    A ``None`` sample does not update the state; the last smoothed value is
    carried forward so consumers never see a gap.
    """
    validate_alpha(alpha)
    sx, sy = EWMA(alpha), EWMA(alpha)
    out: list[tuple[float, float] | None] = []
    for p in points:
        if p is None:
            if sx.value is not None and sy.value is not None:
                out.append((sx.value, sy.value))
            else:
                out.append(None)
        else:
            out.append((sx.update(p[0]), sy.update(p[1])))
    return out
