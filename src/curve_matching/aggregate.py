"""Population-level aggregation across many reference/candidate comparisons.

These functions turn a table of per-pair dissimilarities (as produced by
:mod:`curve_matching.api`) into standardized, uncertainty-weighted scores
comparing several candidates across several references, following equations
17-18 of Bernardi et al.
"""

import numpy as np
import pandas as pd

from .fitting import EPS
from .metrics import METRIC_NAMES

BOXPLOT_STAGES = {"original": "orig", "shift_fraction": "shift", "aligned": "aligned"}


def dimensionless_index(candidate_by_metric):
    """Equation 17, independently for each metric column.

    Parameters
    ----------
    candidate_by_metric : pandas.DataFrame
        One row per candidate, one column per metric.

    Returns
    -------
    standardized : pandas.DataFrame
        ``(value - median) / (IQR / 2)`` for each metric column.
    population : pandas.DataFrame
        The ``median`` and ``IQR`` used, one row per metric.
    """
    table = pd.DataFrame(candidate_by_metric, dtype=float)
    if len(table) < 2:
        raise ValueError("Equation 17 requires a population of at least two candidates")
    median = table.median(axis=0)
    iqr = table.quantile(0.75, axis=0) - table.quantile(0.25, axis=0)
    denominator = iqr / 2.0
    undefined = denominator.abs() <= EPS
    if undefined.any():
        raise ValueError(
            "Equation 17 is undefined for zero-IQR metrics: {}".format(
                ", ".join(table.columns[undefined])
            )
        )
    standardized = (table - median) / denominator
    return standardized, pd.DataFrame({"median": median, "IQR": iqr})


def aggregate_four_metrics(candidate_by_metric):
    """First part of Eq. 18: mean of the four Eq. 17 indices."""
    table = pd.DataFrame(candidate_by_metric).loc[:, METRIC_NAMES]
    standardized, population = dimensionless_index(table)
    return standardized.mean(axis=1), standardized, population


def summarize_comparisons(comparison_table):
    """Calculate the three aggregate indices and epsilon for every curve pair.

    ``comparison_table`` must have ``reference``, ``candidate``, and
    ``metric`` index levels, and ``original``/``shift_fraction``/``aligned`` columns, as produced by :func:`curve_matching.api.compare_tabular_data` or :func:`curve_matching.api.compare_curve_set`.

    The population table reports the equation-17 median and IQR, which are taken across the candidates listed in its ``candidates`` column rather than belonging to any single candidate.

    Returns
    -------
    summary : pandas.DataFrame
        One row per ``(reference, candidate)`` with ``d_hat_original``,
        ``d_hat_shift``, ``d_hat_aligned``, ``epsilon_original``,
        ``epsilon_all``, ``used_original_only``, and ``epsilon``.
    populations : pandas.DataFrame
        Median/IQR/candidate-count per ``(reference, stage, metric)``.
    """
    rows = []
    populations = []
    for reference, group in comparison_table.groupby(level="reference", sort=True):
        shift_wide = group["shift_fraction"].droplevel("reference").unstack("metric")
        use_original_only = bool((shift_wide >= 0.5 - 1e-8).all(axis=0).any())
        values = {}
        for column, label in BOXPLOT_STAGES.items():
            if use_original_only and column != "original":
                values[label] = pd.Series(np.nan, index=values["orig"].index)
                continue
            wide = group[column].droplevel("reference").unstack("metric")
            aggregate, _, population = aggregate_four_metrics(wide)
            values[label] = aggregate
            members = sorted(wide.index.astype(str))
            for metric, stats in population.iterrows():
                populations.append(
                    {
                        "reference": reference,
                        "stage": label,
                        "metric": metric,
                        "median": stats["median"],
                        "IQR": stats["IQR"],
                        "n_candidates": len(members),
                        "candidates": "; ".join(members),
                    }
                )
        for candidate in values["orig"].index:
            row = {
                "reference": reference,
                "candidate": candidate,
                "d_hat_original": values["orig"].loc[candidate],
                "d_hat_shift": values["shift"].loc[candidate],
                "d_hat_aligned": values["aligned"].loc[candidate],
            }
            row["epsilon_original"] = row["d_hat_original"]
            row["epsilon_all"] = (
                np.nan if use_original_only else np.mean(
                    [row["d_hat_original"], row["d_hat_shift"], row["d_hat_aligned"]]
                )
            )
            row["used_original_only"] = use_original_only
            row["epsilon"] = (
                row["epsilon_original"] if use_original_only else row["epsilon_all"]
            )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(populations)


def reference_relative_uncertainty(data, reference_column, y_column, sigma_column):
    """Return max absolute uncertainty / max value for each reference group."""
    required = {reference_column, y_column, sigma_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Input data is missing columns: {}".format(sorted(missing)))
    values = {}
    for reference, group in data.groupby(reference_column):
        sigma = group[sigma_column].dropna().abs()
        magnitude = group[y_column].dropna().abs()
        values[reference] = (
            sigma.max() / magnitude.max()
            if len(sigma) and len(magnitude) and magnitude.max() > 0
            else np.nan
        )
    return pd.Series(values, name="relative_uncertainty", dtype=float)


def weighted_score(epsilon, sigma):
    """Second part of Eq. 18: overall integrated index per candidate.

    Parameters
    ----------
    epsilon : pandas.DataFrame
        Reference-by-candidate matrix of per-pair scores (e.g. the
        ``epsilon`` column of :func:`summarize_comparisons`, pivoted).
    sigma : pandas.Series
        Positive relative uncertainty per reference, aligned to
        ``epsilon``'s index.
    """
    epsilon = pd.DataFrame(epsilon, dtype=float)
    sigma = pd.Series(sigma, index=epsilon.index, dtype=float)
    if sigma.isna().any() or (sigma <= 0.0).any():
        raise ValueError("Every reference requires a positive uncertainty")
    weights = 1.0 / sigma
    return epsilon.mul(weights, axis=0).sum(axis=0) / weights.sum()


def candidate_scores(summary, data, reference_column, y_column, sigma_column, score_column="epsilon"):
    """Build a complete reference-by-candidate score matrix and its weighted total.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`summarize_comparisons`, or any table with
        ``reference``, ``candidate``, and ``score_column`` columns.
    data : pandas.DataFrame
        The original long-form table, used to compute the per-reference
        relative uncertainty via :func:`reference_relative_uncertainty`.
    reference_column, y_column, sigma_column : str
        Column names in ``data`` identifying the reference group, the
        reference measurement, and its uncertainty.

    Returns
    -------
    scores : pandas.DataFrame
        Columns ``candidate`` and ``score``, sorted ascending (lower is
        better agreement), restricted to candidates with a complete row
        across every reference that has a valid uncertainty.
    epsilon : pandas.DataFrame
        The reference-by-candidate matrix actually used.
    sigma : pandas.Series
        The per-reference weights actually used.
    """
    if score_column not in summary:
        raise ValueError("Summary has no score column {!r}".format(score_column))
    epsilon = summary.pivot(index="reference", columns="candidate", values=score_column)
    complete = epsilon.dropna(axis=0, how="any")
    sigma = reference_relative_uncertainty(data, reference_column, y_column, sigma_column)
    sigma = sigma.reindex(complete.index)
    valid = sigma.notna() & (sigma > 0)
    complete, sigma = complete.loc[valid], sigma.loc[valid]
    if complete.empty:
        return pd.DataFrame(columns=["candidate", "score"]), complete, sigma
    scores = weighted_score(complete, sigma).sort_values()
    return scores.rename("score").rename_axis("candidate").reset_index(), complete, sigma


def boxplot_statistics(comparison_table, grouping):
    """Tabulate box statistics of the raw per-pair dissimilarities.

    These are descriptive statistics of the raw dissimilarity measures, not
    the equation-17 normalizer. Use ``grouping="candidate"`` to get, for each
    candidate, the distribution of its dissimilarity across every reference;
    use ``grouping="reference"`` for the reverse. Only the
    ``grouping="reference"`` median and IQR feed equations 17 and 18.
    """
    if grouping not in ("candidate", "reference"):
        raise ValueError("grouping must be 'candidate' or 'reference'")
    required = {"reference", "candidate", "metric"}
    if not required.issubset(comparison_table.index.names):
        raise ValueError("comparison table needs reference, candidate, and metric levels")
    frame = comparison_table.reset_index()
    other = "reference" if grouping == "candidate" else "candidate"
    rows = []
    for column, label in BOXPLOT_STAGES.items():
        for (key, metric), group in frame.groupby([grouping, "metric"], sort=True):
            if metric not in METRIC_NAMES:
                continue
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            rows.append(
                {
                    grouping: key,
                    "stage": label,
                    "metric": metric,
                    "n": int(values.size),
                    "min": values.min(),
                    "Q1": q1,
                    "median": values.median(),
                    "Q3": q3,
                    "max": values.max(),
                    "IQR": q3 - q1,
                    "mean": values.mean(),
                    "aggregated_over": other,
                    "feeds_eq17": grouping == "reference",
                }
            )
    return pd.DataFrame(rows)
