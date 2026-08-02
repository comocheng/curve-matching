"""Dissimilarity metrics and shift alignment between two fitted curves.

The four metrics implemented here (``d_L2_0``, ``d_L2_1``, ``d_P_0``,
``d_P_1``) follow the curve-matching framework of Bernardi et al. (2016),
"Curve matching, a generalized framework for models/experiments comparison"
(equations 4-18), generalized from its original laminar-flame-speed
application to any pair of reference/candidate functional curves. See the
README for the full citation.
"""

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

from .fitting import EPS

METRIC_NAMES = ("d_L2_0", "d_L2_1", "d_P_0", "d_P_1")
METRIC_LABELS = {
    "d_L2_0": r"$d^0_{L^2}$",
    "d_L2_1": r"$d^1_{L^2}$",
    "d_P_0": r"$d^0_P$",
    "d_P_1": r"$d^1_P$",
}


def l2_inner_product(y1, y2, x):
    """Equation 5: the L2 inner product of two curves sampled on ``x``."""
    return float(simpson(np.asarray(y1) * np.asarray(y2), x=x))


def l2_norm(y, x):
    """Equation 4: the L2 norm of a curve sampled on ``x``."""
    return np.sqrt(max(l2_inner_product(y, y, x), 0.0))


def pearson_similarity(y1, y2, x):
    """Equations 11-12: cosine similarity in L2."""
    denominator = l2_norm(y1, x) * l2_norm(y2, x)
    if denominator <= EPS:
        raise ValueError("Pearson distance is undefined for a zero-norm curve")
    return float(np.clip(l2_inner_product(y1, y2, x) / denominator, -1.0, 1.0))


def pearson_distance(y1, y2, x):
    """Equation 13, equivalent to the normalized distances in Eqs. 8-9."""
    return np.sqrt(max((1.0 - pearson_similarity(y1, y2, x)) / 2.0, 0.0))


def common_grid(reference, model, delta=0.0, points=2001):
    """Domain intersection for f(x + delta) and g(x), as in Eq. 16."""
    lower = max(model.xmin, reference.xmin - delta)
    upper = min(model.xmax, reference.xmax - delta)
    if upper <= lower:
        raise ValueError("Shifted curves have no common domain")
    return np.linspace(lower, upper, points)


def curve_dissimilarities(reference, model, delta=0.0, points=2001):
    """Equations 6-9 for f(x + delta) and g(x).

    Parameters
    ----------
    reference, model : FunctionalCurve
        The two curves being compared. ``reference`` is the curve that gets
        shifted by ``delta``.

    Returns
    -------
    dict
        Maps each of :data:`METRIC_NAMES` to a nonnegative dissimilarity.
    """
    x = common_grid(reference, model, delta=delta, points=points)
    # Floating-point endpoints can fall a few ulps outside a spline domain
    # after applying delta. Clipping preserves the mathematical intersection.
    reference_x = np.clip(x + delta, reference.xmin, reference.xmax)
    model_x = np.clip(x, model.xmin, model.xmax)
    f, g = reference(reference_x), model(model_x)
    df = reference(reference_x, derivative=1)
    dg = model(model_x, derivative=1)
    domain_length = x[-1] - x[0]
    amplitude_scale = np.max(np.abs(f))
    derivative_scale = np.max(np.abs(df))
    if amplitude_scale <= EPS or derivative_scale <= EPS:
        raise ValueError("Reference amplitude and derivative scales must be nonzero")
    return {
        "d_L2_0": l2_norm((f - g) / amplitude_scale, x) / domain_length,
        "d_L2_1": l2_norm((df - dg) / derivative_scale, x) / domain_length,
        "d_P_0": pearson_distance(f, g, x),
        "d_P_1": pearson_distance(df, dg, x),
    }


def optimal_shift(reference, model, metric, max_fraction=0.5, points=2001):
    """Equation 16: find the shift of ``reference`` minimizing one metric."""
    if metric not in METRIC_NAMES:
        raise KeyError("Unknown metric {!r}".format(metric))
    limit = max_fraction * reference.span

    def objective(delta):
        return curve_dissimilarities(reference, model, delta=delta, points=points)[metric]

    # The original supplementary implementation searches 1000 equally spaced
    # shifts. A small global scan locates the basin, followed by a bounded
    # refinement and a snap to that publication grid.
    coarse = np.linspace(-limit, limit, 41)
    coarse_values = np.asarray(
        [objective(delta) if -limit <= delta <= limit else np.inf for delta in coarse]
    )
    best = int(np.nanargmin(coarse_values))
    lower = coarse[max(best - 1, 0)]
    upper = coarse[min(best + 1, len(coarse) - 1)]
    result = minimize_scalar(
        objective, bounds=(lower, upper), method="bounded",
        options={"xatol": reference.span * 1e-9},
    )
    if not result.success:
        raise RuntimeError(result.message)
    publication_grid = np.linspace(-limit, limit, 1000)
    nearest = int(np.argmin(np.abs(publication_grid - result.x)))
    candidate_indices = range(max(0, nearest - 2), min(1000, nearest + 3))
    candidates = [(float(publication_grid[index]), objective(publication_grid[index]))
                  for index in candidate_indices]
    return min(candidates, key=lambda item: item[1])


def compare_curves(reference, model, max_fraction=0.5, points=2001):
    """Return original, shift, and aligned dissimilarity for all four metrics.

    Returns
    -------
    pandas.DataFrame
        Indexed by metric name, with columns ``original``, ``delta``,
        ``shift_fraction``, ``shift_percent``, and ``aligned``.
    """
    original = curve_dissimilarities(reference, model, points=points)
    rows = []
    for metric in METRIC_NAMES:
        delta, aligned = optimal_shift(
            reference, model, metric, max_fraction=max_fraction, points=points
        )
        rows.append(
            {
                "metric": metric,
                "original": original[metric],
                "delta": delta,
                "shift_fraction": abs(delta) / reference.span,
                "shift_percent": 100.0 * abs(delta) / reference.span,
                "aligned": aligned,
            }
        )
    return pd.DataFrame(rows).set_index("metric")
