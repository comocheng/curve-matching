"""Domain-agnostic penalized-spline curve fitting and dissimilarity metrics.

Fit reference and candidate curves, measure how much they disagree, and find
the shift that best aligns them. See the README for usage patterns; import
:mod:`curve_matching.plotting` separately for optional matplotlib helpers
(requires the ``plot`` extra).
"""

from ._version import __version__
from .aggregate import (
    BOXPLOT_STAGES,
    aggregate_four_metrics,
    boxplot_statistics,
    candidate_scores,
    dimensionless_index,
    reference_relative_uncertainty,
    summarize_comparisons,
    weighted_score,
)
from .api import compare_curve_data, compare_curve_set, compare_tabular_data
from .fitting import FunctionalCurve, fit_curve
from .metrics import (
    METRIC_LABELS,
    METRIC_NAMES,
    common_grid,
    compare_curves,
    curve_dissimilarities,
    l2_inner_product,
    l2_norm,
    optimal_shift,
    pearson_distance,
    pearson_similarity,
)

__all__ = [
    "__version__",
    "FunctionalCurve",
    "fit_curve",
    "METRIC_NAMES",
    "METRIC_LABELS",
    "BOXPLOT_STAGES",
    "l2_inner_product",
    "l2_norm",
    "pearson_similarity",
    "pearson_distance",
    "common_grid",
    "curve_dissimilarities",
    "optimal_shift",
    "compare_curves",
    "compare_curve_data",
    "compare_curve_set",
    "compare_tabular_data",
    "dimensionless_index",
    "aggregate_four_metrics",
    "summarize_comparisons",
    "reference_relative_uncertainty",
    "weighted_score",
    "candidate_scores",
    "boxplot_statistics",
]
