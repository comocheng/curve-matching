"""Penalized-spline curve fitting."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

EPS = np.finfo(float).eps
INVALID_SCORE = np.finfo(float).max / 1e100


@dataclass(frozen=True)
class FunctionalCurve:
    """A fitted penalized spline over ``[xmin, xmax]``."""

    name: str
    spline: BSpline
    xmin: float
    xmax: float
    smoothing_parameter: float

    @property
    def span(self):
        return self.xmax - self.xmin

    def __call__(self, x, derivative=0):
        return self.spline(x, nu=derivative)


def fit_curve(x, y, name="curve", smoothing=None):
    """Fit a quintic penalized regression spline through scattered ``(x, y)`` data.

    The roughness penalty is the integral of the squared second derivative.
    When ``smoothing`` is ``None``, its coefficient is selected automatically
    by the derivative-based modified GCV criterion, so the fit needs no
    domain-specific tuning.

    Parameters
    ----------
    x, y : array-like
        One-dimensional, equal-length samples. Non-finite entries are
        dropped and the remaining points are sorted by ``x``. At least three
        distinct ``x`` values are required.
    name : str
        Label carried through onto the returned curve for use in plots and
        result tables.
    smoothing : float, optional
        Fixed roughness-penalty coefficient. If omitted, it is selected
        automatically.

    Returns
    -------
    FunctionalCurve
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("At least three distinct x values are required")
    degree = min(5, len(x) - 1)
    basis_count = min(len(x), 30)
    break_count = max(basis_count - degree + 1, 2)
    breaks = np.unique(np.quantile(x, np.linspace(0.0, 1.0, break_count)))
    knots = np.r_[
        np.repeat(breaks[0], degree + 1),
        breaks[1:-1],
        np.repeat(breaks[-1], degree + 1),
    ]
    basis_count = len(knots) - degree - 1
    basis = BSpline(knots, np.eye(basis_count), degree, extrapolate=False)
    design = basis(x)
    penalty = _roughness_penalty(basis, breaks, derivative=2)
    cross_product = design.T @ design
    right_hand_side = design.T @ y

    derivative_x = x[1:-1]
    observed_derivative = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    derivative_design = basis(derivative_x, nu=1)

    def solve(log10_lambda, return_coefficients=False):
        smoothing_parameter = 10.0 ** float(log10_lambda)
        system = cross_product + smoothing_parameter * penalty
        try:
            coefficients = np.linalg.solve(system, right_hand_side)
            influence = np.linalg.solve(system, design.T)
        except np.linalg.LinAlgError:
            return (INVALID_SCORE, None) if return_coefficients else INVALID_SCORE
        effective_df = float(np.trace(design @ influence))
        denominator = (len(x) - effective_df) ** 2
        if denominator <= EPS:
            score = INVALID_SCORE
        else:
            residual = observed_derivative - derivative_design @ coefficients
            score = float(len(x) * np.sum(residual ** 2) / denominator)
        if not np.isfinite(score):
            score = INVALID_SCORE
        return (score, coefficients) if return_coefficients else score

    if smoothing is None:
        candidates = np.linspace(-16.0, 8.0, 241)
        scores = np.asarray([solve(value) for value in candidates])
        best = int(np.argmin(scores))
        lower = candidates[max(best - 1, 0)]
        upper = candidates[min(best + 1, len(candidates) - 1)]
        if lower == upper:
            log10_lambda = float(candidates[best])
        else:
            optimum = minimize_scalar(solve, bounds=(lower, upper), method="bounded")
            log10_lambda = float(optimum.x if optimum.success else candidates[best])
        smoothing_parameter = 10.0 ** log10_lambda
    else:
        smoothing_parameter = float(smoothing)
        if smoothing_parameter < 0.0:
            raise ValueError("smoothing must be nonnegative or None")
        log10_lambda = np.log10(max(smoothing_parameter, 1e-16))

    _, coefficients = solve(log10_lambda, return_coefficients=True)
    if coefficients is None:
        raise RuntimeError("Could not solve the penalized-spline system")
    spline = BSpline(knots, coefficients, degree, extrapolate=False)
    return FunctionalCurve(
        name, spline, float(x[0]), float(x[-1]), smoothing_parameter
    )


def _roughness_penalty(basis, breaks, derivative=2, quadrature_points=12):
    """Integrate products of B-spline derivatives over the curve domain."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    basis_count = basis.c.shape[0]
    penalty = np.zeros((basis_count, basis_count))
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        x = (lower + upper) / 2.0 + (upper - lower) * nodes / 2.0
        derivatives = basis(x, nu=derivative)
        penalty += (
            (derivatives.T * weights) @ derivatives * (upper - lower) / 2.0
        )
    return penalty
