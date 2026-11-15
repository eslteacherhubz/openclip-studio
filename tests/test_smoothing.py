"""Unit tests for :mod:`openclip.ai.smoothing` (EWMA, moving average, optional pairs).

Fully offline: synthetic series only, deterministic pseudo-random jitter from
a fixed seed.
"""

from __future__ import annotations

import math
import random

import pytest

from openclip.ai.smoothing import (
    EWMA,
    moving_average_series,
    smooth_optional_pairs,
    smooth_series,
    validate_alpha,
)


def _noisy_series(n: int = 200, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    return [math.sin(2.0 * math.pi * i / 50.0) + rng.uniform(-0.3, 0.3) for i in range(n)]


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class TestEWMA:
    def test_variance_reduction_on_noisy_series(self) -> None:
        raw = _noisy_series()
        ewma = EWMA(0.3)
        smoothed = [ewma.update(v) for v in raw]
        assert _variance(smoothed) < _variance(raw)

    def test_first_sample_initializes_state(self) -> None:
        ewma = EWMA(0.5)
        assert ewma.update(2.0) == 2.0
        assert ewma.update(4.0) == 3.0

    def test_reset_clears_state(self) -> None:
        ewma = EWMA(0.5)
        ewma.update(2.0)
        ewma.reset()
        assert ewma.value is None
        assert ewma.update(10.0) == 10.0


class TestSmoothSeries:
    def test_output_length_equals_input_length(self) -> None:
        values = _noisy_series(n=50)
        assert len(smooth_series(values)) == len(values)

    def test_first_sample_initializes(self) -> None:
        out = smooth_series([5.0, 1.0, 9.0], alpha=0.3)
        assert out[0] == 5.0

    def test_alpha_one_is_passthrough(self) -> None:
        values = _noisy_series(n=30, seed=7)
        assert smooth_series(values, alpha=1.0) == values


class TestMovingAverageSeries:
    def test_hand_computed_window_3(self) -> None:
        assert moving_average_series([1.0, 2.0, 3.0, 4.0], window=3) == [1.0, 1.5, 2.0, 3.0]

    def test_hand_computed_window_2(self) -> None:
        assert moving_average_series([1.0, 2.0, 3.0, 4.0], window=2) == [1.0, 1.5, 2.5, 3.5]

    def test_window_1_is_passthrough(self) -> None:
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert moving_average_series(values, window=1) == values

    def test_window_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            moving_average_series([1.0, 2.0], window=0)
        with pytest.raises(ValueError):
            moving_average_series([1.0, 2.0], window=-3)


class TestSmoothOptionalPairs:
    def test_dropout_carries_last_smoothed_value_forward(self) -> None:
        points: list[tuple[float, float] | None] = [(1.0, 1.0), None, None, (3.0, 3.0), None]
        out = smooth_optional_pairs(points, alpha=0.3)
        assert out[0] == (1.0, 1.0)
        assert out[1] == (1.0, 1.0)
        assert out[2] == (1.0, 1.0)
        assert out[3] == pytest.approx((1.6, 1.6))
        assert out[4] == pytest.approx((1.6, 1.6))

    def test_no_gaps_after_initialization(self) -> None:
        points: list[tuple[float, float] | None] = [(0.5, 0.5)] + [None] * 10
        out = smooth_optional_pairs(points)
        assert all(p is not None for p in out)
        assert all(p == (0.5, 0.5) for p in out)

    def test_leading_nones_stay_none(self) -> None:
        out = smooth_optional_pairs([None, None, (2.0, 4.0)], alpha=1.0)
        assert out[0] is None
        assert out[1] is None
        assert out[2] == (2.0, 4.0)

    def test_alpha_one_passthrough_with_dropout_hold(self) -> None:
        points: list[tuple[float, float] | None] = [(1.0, 2.0), None, (5.0, 6.0), None]
        out = smooth_optional_pairs(points, alpha=1.0)
        assert out == [(1.0, 2.0), (1.0, 2.0), (5.0, 6.0), (5.0, 6.0)]


class TestValidateAlpha:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(-0.5)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(1.5)

    def test_valid_values_returned(self) -> None:
        assert validate_alpha(1.0) == 1.0
        assert validate_alpha(0.3) == 0.3
