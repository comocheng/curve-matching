"""High-level entry points for comparing reference and candidate data."""

import numpy as np
import pandas as pd

from .fitting import fit_curve
from .metrics import compare_curves


def compare_curve_data(
    x_reference,
    y_reference,
    x_candidate,
    y_candidate,
    reference_name="reference",
    candidate_name="candidate",
    smoothing=None,
    max_fraction=0.5,
    points=2001,
):
    """Fit and compare a single reference curve against a single candidate curve.

    ``x_reference``/``y_reference`` and ``x_candidate``/``y_candidate`` may use
    different sampling grids and different numbers of points; each is fit with
    its own penalized spline before the two are compared.

    Returns
    -------
    pandas.DataFrame
        Indexed by metric name, with columns ``original``, ``delta``,
        ``shift_fraction``, ``shift_percent``, and ``aligned`` (see
        :func:`curve_matching.metrics.compare_curves`).
    """
    reference_curve = fit_curve(x_reference, y_reference, name=reference_name, smoothing=smoothing)
    candidate_curve = fit_curve(x_candidate, y_candidate, name=candidate_name, smoothing=smoothing)
    return compare_curves(reference_curve, candidate_curve, max_fraction=max_fraction, points=points)


def compare_curve_set(
    x_reference,
    y_reference,
    candidates,
    reference_name="reference",
    smoothing=None,
    max_fraction=0.5,
    points=2001,
):
    """Compare one reference curve against several named candidates.

    Parameters
    ----------
    x_reference, y_reference : array-like
        The reference data.
    candidates : dict[str, tuple[array-like, array-like]]
        Maps a candidate name to its own ``(x, y)`` data.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(candidate, metric)``, with the same columns as
        :func:`compare_curve_data`.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one named (x, y) pair")
    reference_curve = fit_curve(x_reference, y_reference, name=reference_name, smoothing=smoothing)
    comparisons = {}
    for name, (x_candidate, y_candidate) in candidates.items():
        candidate_curve = fit_curve(x_candidate, y_candidate, name=name, smoothing=smoothing)
        comparisons[name] = compare_curves(
            reference_curve, candidate_curve, max_fraction=max_fraction, points=points
        )
    return pd.concat(comparisons, names=["candidate", "metric"])


def compare_tabular_data(
    data,
    reference_column,
    candidate_column,
    reference_x_column,
    reference_y_column,
    candidate_x_column,
    candidate_y_column,
    order_column=None,
    reference_mask_column=None,
    min_points=3,
    smoothing=None,
    max_fraction=0.5,
    points=2001,
):
    """Fit and compare every reference/candidate pair found in a long-form table.

    ``data`` has one row per observation, grouped by ``reference_column`` and
    ``candidate_column``. Within each group, rows are (optionally) sorted by
    ``order_column``, then split into the reference series
    (``reference_x_column``, ``reference_y_column``) and the candidate series
    (``candidate_x_column``, ``candidate_y_column``); non-finite values and
    duplicate x-values are dropped from each series independently before
    fitting.

    If ``reference_mask_column`` is given, only rows where it is truthy
    contribute to the reference series -- useful when a table interleaves
    genuine reference measurements with rows that exist only to carry a
    candidate value at an extra grid point. When omitted, every row with a
    finite reference x/y pair is used.

    A group is skipped (and recorded with a reason in ``coverage``) when
    either series has fewer than ``min_points`` usable rows, or when fitting
    raises an error.

    Returns
    -------
    comparisons : pandas.DataFrame
        Indexed by ``(reference, candidate, metric)``, with the columns
        produced by :func:`curve_matching.metrics.compare_curves`.
    coverage : pandas.DataFrame
        One row per ``(reference, candidate)`` group, reporting point counts,
        ``status`` (``"matched"`` or ``"skipped"``), and ``reason``.
    """
    required = {
        reference_column, candidate_column,
        reference_x_column, reference_y_column,
        candidate_x_column, candidate_y_column,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Input data is missing columns: {}".format(sorted(missing)))

    comparisons = {}
    coverage = []
    for (reference, candidate), group in data.groupby(
        [reference_column, candidate_column], sort=True
    ):
        ordered = group.sort_values(order_column) if order_column else group
        if reference_mask_column and reference_mask_column in ordered:
            reference_mask = ordered[reference_mask_column].fillna(True).astype(bool)
        else:
            reference_mask = pd.Series(True, index=ordered.index)
        reference_rows = ordered.loc[
            reference_mask
            & np.isfinite(ordered[reference_x_column])
            & np.isfinite(ordered[reference_y_column])
        ].drop_duplicates(reference_x_column)
        candidate_rows = ordered.loc[
            np.isfinite(ordered[candidate_x_column])
            & np.isfinite(ordered[candidate_y_column])
        ].drop_duplicates(candidate_x_column)

        reason = ""
        if len(reference_rows) < min_points:
            reason = "fewer than {} usable reference points".format(min_points)
        elif len(candidate_rows) < min_points:
            reason = "fewer than {} usable candidate points".format(min_points)
        else:
            try:
                reference_curve = fit_curve(
                    reference_rows[reference_x_column], reference_rows[reference_y_column],
                    name=str(reference), smoothing=smoothing,
                )
                candidate_curve = fit_curve(
                    candidate_rows[candidate_x_column], candidate_rows[candidate_y_column],
                    name=str(candidate), smoothing=smoothing,
                )
                comparisons[(reference, candidate)] = compare_curves(
                    reference_curve, candidate_curve, max_fraction=max_fraction, points=points,
                )
            except Exception as exc:
                reason = "{}: {}".format(type(exc).__name__, exc)
        coverage.append(
            {
                "reference": reference,
                "candidate": candidate,
                "points_reference": len(reference_rows),
                "points_candidate": len(candidate_rows),
                "status": "matched" if (reference, candidate) in comparisons else "skipped",
                "reason": reason,
            }
        )

    if comparisons:
        table = pd.concat(comparisons, names=["reference", "candidate", "metric"])
    else:
        table = pd.DataFrame(
            columns=["original", "delta", "shift_fraction", "shift_percent", "aligned"]
        )
        table.index = pd.MultiIndex.from_arrays(
            [[], [], []], names=["reference", "candidate", "metric"]
        )
    return table, pd.DataFrame(coverage)
