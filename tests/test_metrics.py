import numpy as np
import pytest

from curve_matching import METRIC_NAMES, compare_curves, curve_dissimilarities, fit_curve


def test_identical_curves_have_zero_original_distance():
    x = np.linspace(0.7, 1.4, 8)
    y = 40.0 - 80.0 * (x - 1.1) ** 2
    curve = fit_curve(x, y)
    result = compare_curves(curve, curve)
    assert set(result.index) == set(METRIC_NAMES)
    assert np.allclose(result["original"], 0.0, atol=1e-7)


def test_quintic_gcv_matches_published_case_16_original_metrics():
    # Reference values from Bernardi et al.'s case-16 laminar-flame-speed
    # supplementary case study; used here purely as a numeric regression
    # fixture for the generic spline fit and dissimilarity metrics.
    x_reference = np.array([0.7, 0.801, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    y_reference = np.array([21.4, 28.2, 36.5, 41.5, 43.6, 42.2, 34.9, 29.4])
    x_candidate = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4])
    y_candidate = np.array([22.0, 29.3, 35.6, 38.0, 39.8, 40.8, 41.1, 40.6, 39.2, 34.2, 26.9])
    published = {
        "d_L2_0": 0.0456630660864495,
        "d_L2_1": 0.108408370393915,
        "d_P_0": 0.0131860540867428,
        "d_P_1": 0.0481902314071151,
    }
    calculated = curve_dissimilarities(
        fit_curve(x_reference, y_reference),
        fit_curve(x_candidate, y_candidate),
    )
    for metric, value in published.items():
        assert calculated[metric] == pytest.approx(value, rel=0.06)


def test_shifted_domain_endpoints_are_clipped_to_the_intersection():
    x = np.linspace(0.7, 1.4, 8)
    reference = fit_curve(x, 40.0 - 80.0 * (x - 1.1) ** 2)
    candidate = fit_curve(x, 39.0 - 70.0 * (x - 1.05) ** 2)
    values = curve_dissimilarities(reference, candidate, delta=0.1)
    assert all(np.isfinite(value) for value in values.values())


def test_unknown_metric_is_rejected():
    from curve_matching import optimal_shift

    x = np.linspace(0.0, 1.0, 10)
    curve = fit_curve(x, np.sin(2.0 * np.pi * x))
    with pytest.raises(KeyError):
        optimal_shift(curve, curve, "not_a_metric")
