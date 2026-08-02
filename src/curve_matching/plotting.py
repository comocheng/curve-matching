"""Optional matplotlib visualizations for curve fits, shifts, and scores.

Requires the ``plot`` extra: ``pip install curve-matching[plot]``. Nothing
else in the package depends on matplotlib, so this module is only imported
when you explicitly do ``from curve_matching import plotting`` or
``import curve_matching.plotting``.
"""

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "curve_matching.plotting requires matplotlib; install with "
        "`pip install curve-matching[plot]`"
    ) from exc

from .aggregate import BOXPLOT_STAGES
from .fitting import fit_curve
from .metrics import METRIC_LABELS, METRIC_NAMES, curve_dissimilarities, optimal_shift

SHIFT_PLOT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_curve_fits(
    data,
    reference_column,
    candidate_column,
    x_column,
    reference_y_column,
    candidate_y_column,
    order_column=None,
    reference_mask_column=None,
    sigma_column=None,
    show_fits=True,
    smoothing=None,
    min_points=3,
    grid_points=400,
    columns=4,
    colors=None,
    xlabel="x",
    ylabel="y",
    title="Curve fits",
):
    """Plot every reference's measured points against every candidate's, one panel per reference.

    Set ``show_fits=False`` to draw only the raw points; by default the
    penalized-spline fit (as used by the dissimilarity metrics) is overlaid
    for both the reference and each candidate.
    """
    required = {reference_column, candidate_column, x_column, reference_y_column, candidate_y_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Input data is missing columns: {}".format(sorted(missing)))
    references = list(dict.fromkeys(data[reference_column]))
    candidates = list(dict.fromkeys(data[candidate_column]))
    if not references or not candidates:
        raise ValueError("No data available to plot")

    columns = max(1, min(int(columns), len(references)))
    rows = int(np.ceil(len(references) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.5 * columns, max(3.1 * rows, 3.4)),
        sharex=False, sharey=False, squeeze=False,
    )
    if colors is None:
        colors = plt.colormaps["tab10"](np.linspace(0.0, 0.9, max(len(candidates), 2)))
    elif len(colors) < len(candidates):
        raise ValueError("{} hues supplied for {} candidates".format(len(colors), len(candidates)))
    candidate_colors = dict(zip(candidates, colors))

    def _fit(points, value_column, name):
        usable = points.loc[
            np.isfinite(points[x_column]) & np.isfinite(points[value_column])
        ].drop_duplicates(x_column).sort_values(x_column)
        if not show_fits or len(usable) < min_points:
            return usable, None
        try:
            return usable, fit_curve(usable[x_column], usable[value_column], name=name, smoothing=smoothing)
        except Exception:
            return usable, None

    for axis, reference in zip(axes.flat, references):
        subset = data.loc[data[reference_column] == reference]
        if order_column and order_column in subset:
            subset = subset.sort_values(order_column)
        if reference_mask_column and reference_mask_column in subset:
            reference_mask = subset[reference_mask_column].fillna(True).astype(bool)
        else:
            reference_mask = pd.Series(True, index=subset.index)
        measured = subset.loc[reference_mask]
        if order_column:
            measured = measured.drop_duplicates(order_column)
        ref_points, ref_curve = _fit(measured, reference_y_column, str(reference))
        yerr = ref_points[sigma_column] if sigma_column and sigma_column in ref_points else None
        axis.errorbar(
            ref_points[x_column], ref_points[reference_y_column], yerr=yerr,
            linestyle="none", marker="s", markersize=4.0, color="black",
            markerfacecolor="none", markeredgewidth=0.9,
            capsize=2.0, elinewidth=0.8, label="reference", zorder=4,
        )
        if ref_curve is not None:
            grid = np.linspace(ref_curve.xmin, ref_curve.xmax, grid_points)
            axis.plot(grid, ref_curve(grid), color="black", linewidth=1.6, zorder=4)
        for candidate in candidates:
            candidate_points = subset.loc[subset[candidate_column] == candidate]
            points, curve = _fit(candidate_points, candidate_y_column, str(candidate))
            if not points.empty:
                axis.plot(
                    points[x_column], points[candidate_y_column], linestyle="none",
                    marker=".", markersize=3.0, color=candidate_colors[candidate],
                    alpha=0.55 if show_fits else 1.0, zorder=3,
                )
            if curve is not None:
                grid = np.linspace(curve.xmin, curve.xmax, grid_points)
                axis.plot(grid, curve(grid), color=candidate_colors[candidate], linewidth=1.3, zorder=3)
        axis.set_title(str(reference), fontsize=9)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.tick_params(labelsize=8)
        legend_handles = [
            plt.Line2D(
                [], [], color="black", marker="s", markerfacecolor="none",
                linestyle="none", markersize=5.0, label="reference",
            )
        ]
        legend_handles += [
            plt.Line2D([], [], color=candidate_colors[c], linewidth=1.5, label=str(c))
            for c in candidates
        ]
        axis.legend(
            handles=legend_handles, fontsize=6, loc="best", frameon=True,
            framealpha=0.85, handlelength=1.6, labelspacing=0.3, borderpad=0.3,
        )
        axis.set_xlabel(xlabel, fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
    for axis in axes.flat[len(references):]:
        axis.set_visible(False)

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure, axes


def plot_shift_alignment(
    data,
    reference_column,
    candidate_column,
    x_column,
    reference_y_column,
    candidate_y_column,
    order_column=None,
    reference_mask_column=None,
    sigma_column=None,
    references=None,
    candidates=None,
    metric="d_L2_0",
    smoothing=None,
    min_points=3,
    grid_points=400,
    max_fraction=0.5,
    colors=None,
    xlabel="x",
    ylabel="y",
    title=None,
):
    """Plot original and shifted candidate curves against each reference.

    For every requested reference, the measured points and their reference
    spline are drawn once, then each candidate contributes a solid curve for
    its original fit and a dashed curve translated by the optimal shift of
    ``metric``. Because Eq. 16 compares ``reference(x + delta)`` with
    ``candidate(x)``, the aligned pairing is displayed by translating the
    candidate curve by ``delta`` and leaving the reference in place.

    Returns the figure, axes, and a table reporting each candidate's shift and
    the resulting drop in dissimilarity.
    """
    if metric not in METRIC_NAMES:
        raise KeyError("Unknown metric {!r}".format(metric))
    required = {reference_column, candidate_column, x_column, reference_y_column, candidate_y_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Input data is missing columns: {}".format(sorted(missing)))

    frame = data
    if references is not None:
        references = [references] if isinstance(references, str) else list(references)
        frame = frame.loc[frame[reference_column].isin(references)]
    else:
        references = list(dict.fromkeys(frame[reference_column]))
    if candidates is not None:
        candidates = [candidates] if isinstance(candidates, str) else list(candidates)
        frame = frame.loc[frame[candidate_column].isin(candidates)]
    else:
        candidates = list(dict.fromkeys(frame[candidate_column]))
    if not references or not candidates:
        raise ValueError("No data matches the requested references and candidates")

    palette = tuple(colors) if colors else SHIFT_PLOT_COLORS
    if len(candidates) > len(palette):
        raise ValueError(
            "{} candidates requested but only {} separable hues are defined; split "
            "them across figures or pass an explicit palette".format(len(candidates), len(palette))
        )
    candidate_colors = dict(zip(candidates, palette))

    def _fit(points, value_column, name):
        usable = points.loc[
            np.isfinite(points[x_column]) & np.isfinite(points[value_column])
        ].drop_duplicates(x_column).sort_values(x_column)
        if len(usable) < min_points:
            return usable, None
        try:
            return usable, fit_curve(usable[x_column], usable[value_column], name=name, smoothing=smoothing)
        except Exception:
            return usable, None

    columns = min(len(references), 2)
    rows = int(np.ceil(len(references) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.6 * columns, 5.0 * rows), squeeze=False)
    records = []

    for axis, reference in zip(axes.flat, references):
        subset = frame.loc[frame[reference_column] == reference]
        if order_column and order_column in subset:
            subset = subset.sort_values(order_column)
        if reference_mask_column and reference_mask_column in subset:
            reference_mask = subset[reference_mask_column].fillna(True).astype(bool)
        else:
            reference_mask = pd.Series(True, index=subset.index)
        measured = subset.loc[reference_mask]
        if order_column:
            measured = measured.drop_duplicates(order_column)
        ref_points, reference_curve = _fit(measured, reference_y_column, str(reference))
        yerr = ref_points[sigma_column] if sigma_column and sigma_column in ref_points else None
        handles = [
            axis.errorbar(
                ref_points[x_column], ref_points[reference_y_column], yerr=yerr,
                linestyle="none", marker="o", markersize=8.0, color="#1a1a19",
                capsize=2.5, elinewidth=0.9, zorder=6, label="Reference",
            )
        ]
        if reference_curve is not None:
            grid = np.linspace(reference_curve.xmin, reference_curve.xmax, grid_points)
            handles += axis.plot(
                grid, reference_curve(grid), color="#1a1a19", linewidth=1.1,
                alpha=0.55, zorder=5, label="Reference, spline",
            )

        for candidate in candidates:
            points, curve = _fit(
                subset.loc[subset[candidate_column] == candidate], candidate_y_column, str(candidate)
            )
            if curve is None:
                continue
            grid = np.linspace(curve.xmin, curve.xmax, grid_points)
            handles += axis.plot(
                grid, curve(grid), color=candidate_colors[candidate], linewidth=2.0,
                zorder=4, label=str(candidate),
            )
            if reference_curve is None:
                continue
            original = curve_dissimilarities(reference_curve, curve)[metric]
            delta, aligned = optimal_shift(reference_curve, curve, metric, max_fraction=max_fraction)
            percent = 100.0 * abs(delta) / reference_curve.span
            handles += axis.plot(
                grid + delta, curve(grid), color=candidate_colors[candidate],
                linewidth=2.0, linestyle=(0, (5, 2)), zorder=4,
                label="{} shifted, {:.2f}%".format(candidate, percent),
            )
            records.append(
                {
                    "reference": reference, "candidate": candidate, "metric": metric,
                    "original": original, "delta": delta,
                    "shift_percent": percent, "aligned": aligned,
                    "improvement": original - aligned,
                }
            )

        if len(references) > 1:
            axis.set_title(str(reference), fontsize=11)
        axis.set_xlabel(xlabel, fontsize=10)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(
            handles=handles, fontsize=8, loc="lower center", frameon=True,
            framealpha=0.9, handlelength=2.4, labelspacing=0.35,
            title="optimal shift of {}".format(METRIC_LABELS[metric]), title_fontsize=8,
        )
    for axis in axes.flat[len(references):]:
        axis.set_visible(False)

    if title is None:
        label = str(references[0]) if len(references) == 1 else "curve alignment"
        title = "{}: original (solid) and shifted (dashed) candidate curves".format(label)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure, axes, pd.DataFrame(records)


def _plot_boxplots(comparison_table, grouping, title):
    required = {"reference", "candidate", "metric"}
    if not required.issubset(comparison_table.index.names):
        raise ValueError("comparison table needs reference, candidate, and metric levels")
    frame = comparison_table.reset_index()
    labels = list(dict.fromkeys(frame[grouping]))
    if not labels:
        raise ValueError("No curve comparisons are available to plot")
    colors = plt.colormaps["tab20"](np.linspace(0.05, 0.95, max(len(labels), 2)))
    fig, axes = plt.subplots(
        len(METRIC_NAMES), len(BOXPLOT_STAGES),
        figsize=(max(11.0, 0.65 * len(labels) + 7.0), 11.5), squeeze=False,
    )
    for row, metric in enumerate(METRIC_NAMES):
        metric_frame = frame.loc[frame["metric"] == metric]
        for column, (value_column, stage_label) in enumerate(BOXPLOT_STAGES.items()):
            ax = axes[row, column]
            distributions = [
                metric_frame.loc[metric_frame[grouping] == label, value_column]
                .dropna().to_numpy(dtype=float)
                for label in labels
            ]
            artists = ax.boxplot(
                distributions, patch_artist=True, widths=0.62, showfliers=True,
                medianprops={"color": "black", "linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 3.5,
                            "markerfacecolor": "none", "markeredgecolor": "0.25"},
            )
            for box, color in zip(artists["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.88)
            ax.set_title("{} ({})".format(METRIC_LABELS[metric], stage_label))
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels([str(label) for label in labels] if row == len(METRIC_NAMES) - 1 else [])
            if row == len(METRIC_NAMES) - 1:
                ax.tick_params(axis="x", labelrotation=45)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                ax.set_xlabel(
                    "{} (N={})".format(grouping.capitalize() + "s", len(labels)),
                    fontsize=8,
                )
            ax.set_ylabel("dissimilarity / shift fraction", fontsize=8)
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.96))
    return fig, axes


def plot_reference_boxplots(comparison_table, title="Curve matching by reference"):
    """Plot candidate-population distributions, grouped by reference."""
    return _plot_boxplots(comparison_table, "reference", title)


def plot_candidate_boxplots(comparison_table, title="Curve matching by candidate"):
    """Plot reference-population distributions, grouped by candidate."""
    return _plot_boxplots(comparison_table, "candidate", title)


def plot_candidate_scores(scores, score_column="score", title="Integrated curve-matching score"):
    """Plot the equation-18 scores from :func:`curve_matching.aggregate.candidate_scores`.

    Lower values indicate better agreement with the reference data.
    """
    table = pd.DataFrame(scores).sort_values(score_column, ascending=True)
    if table.empty or not {"candidate", score_column}.issubset(table.columns):
        raise ValueError(
            "Candidate scores require nonempty candidate and {!r} columns".format(score_column)
        )
    height = max(3.4, 0.55 * len(table) + 1.5)
    figure, axis = plt.subplots(figsize=(7.2, height))
    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.8, len(table)))
    axis.barh(
        table["candidate"].astype(str), table[score_column],
        color=colors, edgecolor="0.2", linewidth=0.5,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Integrated score (lower is better)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.22, linewidth=0.6)
    for position, value in enumerate(table[score_column]):
        axis.text(value, position, "  {:.3g}".format(value), va="center", fontsize=8)
    figure.tight_layout()
    return figure, axis


def plot_metric_bars(comparison_table, stage="original", title="Curve-matching dissimilarity measures"):
    """Plot the four dissimilarity measures per reference and candidate.

    ``stage`` selects the ``original`` (unshifted) or ``aligned`` (optimally
    shifted) dissimilarity column. Unlike the Eq. 17-18 boxplots, this
    visualization works for a single candidate.
    """
    required = {"reference", "candidate", "metric"}
    if not required.issubset(comparison_table.index.names):
        raise ValueError("comparison table needs reference, candidate, and metric levels")
    if stage not in comparison_table.columns:
        raise ValueError("Unknown stage column {!r}".format(stage))
    frame = comparison_table.reset_index()
    references = list(dict.fromkeys(frame["reference"]))
    candidates = list(dict.fromkeys(frame["candidate"]))
    if not references or not candidates:
        raise ValueError("No curve comparisons are available to plot")

    colors = plt.colormaps["tab10"](np.linspace(0.0, 0.9, max(len(candidates), 2)))
    candidate_colors = dict(zip(candidates, colors))
    figure, axes = plt.subplots(
        2, 2, figsize=(max(11.0, 0.5 * len(references) + 7.0), 8.5), squeeze=False
    )
    positions = np.arange(len(references))
    width = 0.8 / max(len(candidates), 1)
    for index, metric in enumerate(METRIC_NAMES):
        axis = axes[index // 2][index % 2]
        metric_frame = frame.loc[frame["metric"] == metric]
        lookup = metric_frame.set_index(["reference", "candidate"])[stage]
        for offset, candidate in enumerate(candidates):
            heights = [
                float(lookup.get((reference, candidate), np.nan))
                for reference in references
            ]
            axis.bar(
                positions + (offset - (len(candidates) - 1) / 2.0) * width,
                heights, width=width, color=candidate_colors[candidate],
                edgecolor="0.2", linewidth=0.4, label=str(candidate),
            )
        axis.set_title("{} ({})".format(METRIC_LABELS[metric], stage))
        axis.set_xticks(positions)
        axis.set_xticklabels([str(reference) for reference in references], rotation=45, ha="right", fontsize=7)
        axis.set_xlabel("reference", fontsize=8)
        axis.set_ylabel("{} dissimilarity".format(METRIC_LABELS[metric]), fontsize=8)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    handles = [
        plt.Line2D([], [], color=candidate_colors[c], linewidth=6, label=str(c))
        for c in candidates
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=min(4, len(handles)),
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.02, 0.06, 1.0, 0.96))
    return figure, axes
