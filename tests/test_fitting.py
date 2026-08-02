import numpy as np
import pytest

from curve_matching import fit_curve


def test_fit_curve_requires_at_least_three_distinct_x_values():
    with pytest.raises(ValueError, match="At least three distinct"):
        fit_curve([0.0, 1.0], [0.0, 1.0])


def test_fit_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one-dimensional arrays of equal length"):
        fit_curve([0.0, 1.0, 2.0], [0.0, 1.0])


def test_fit_curve_recovers_a_smooth_parabola():
    x = np.linspace(0.0, 1.0, 25)
    y = 1.0 - (x - 0.5) ** 2
    curve = fit_curve(x, y)
    assert curve.xmin == pytest.approx(0.0)
    assert curve.xmax == pytest.approx(1.0)
    assert curve.span == pytest.approx(1.0)
    grid = np.linspace(0.05, 0.95, 50)
    assert np.allclose(curve(grid), 1.0 - (grid - 0.5) ** 2, atol=5e-3)


def test_fit_curve_accepts_a_fixed_smoothing_parameter():
    x = np.linspace(0.0, 1.0, 25)
    y = np.sin(2.0 * np.pi * x) + 0.01 * np.random.RandomState(0).randn(25)
    curve = fit_curve(x, y, smoothing=1e-3)
    assert curve.smoothing_parameter == pytest.approx(1e-3)


def test_fit_curve_rejects_negative_smoothing():
    with pytest.raises(ValueError, match="nonnegative"):
        fit_curve(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10), smoothing=-1.0)
