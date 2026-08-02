# curve-matching

Fit noisy, scattered data with a penalized regression spline, then measure
how much a reference curve and one or more candidate curves disagree, and
find the shift that best aligns them.

This package implements the curve-matching framework of Bernardi et al.
(2016) [1] (equations 4-18): a quintic penalized spline with automatic smoothing
selection (GCV), four dissimilarity metrics (amplitude and derivative
versions of an L2 distance and a Pearson/cosine distance), an optimal-shift
search, and a robust, uncertainty-weighted aggregate score across many
reference/candidate pairs.

It has no notion of what your curves represent. Anywhere a specific
application (flame speeds, reaction rates, sensor traces, model benchmarks,
...) needs its own column names, axis labels, or grouping keys, that is a
parameter you pass in, not something baked into the code.

## Installation

Directly from GitHub:

```bash
pip install git+https://github.com/comocheng/curve-matching.git
```

Or clone and install locally:

```bash
git clone https://github.com/comocheng/curve-matching.git
cd curve-matching
pip install .
```

Once published to PyPI:

```bash
pip install curve-matching
```

Plotting helpers are an optional extra (they pull in matplotlib) -- append
`[plot]` to any of the install commands above, e.g.:

```bash
pip install "curve-matching[plot] @ git+https://github.com/comocheng/curve-matching.git"
```

## Quickstart

### Compare two curves directly

Reference and candidate data may use different sampling grids and different
numbers of points:

```python
import numpy as np
from curve_matching import compare_curve_data

x_reference = np.linspace(0.0, 1.0, 12)
y_reference = np.sin(2.0 * np.pi * x_reference)
x_candidate = np.linspace(0.0, 1.0, 20)
y_candidate = np.sin(2.0 * np.pi * x_candidate) + 0.02

comparison = compare_curve_data(
    x_reference, y_reference,
    x_candidate, y_candidate,
    reference_name="measurement",
    candidate_name="simulation",
)
```

`comparison` is a `DataFrame` indexed by metric name (`d_L2_0`, `d_L2_1`,
`d_P_0`, `d_P_1`), with columns `original`, `delta`, `shift_fraction`,
`shift_percent`, and `aligned`.

### Compare one reference against several candidates

```python
from curve_matching import compare_curve_set

comparison = compare_curve_set(
    x_reference, y_reference,
    {
        "model-a": (x_a, y_a),
        "model-b": (x_b, y_b),
    },
)
```

### Compare every pair in a long-form table

For tabular results where every row is one observation, all schema-dependent
column names are explicit arguments.

```python
from curve_matching import compare_tabular_data

comparison, coverage = compare_tabular_data(
    results,
    reference_column="data_set",
    candidate_column="algorithm",
    reference_x_column="measured_time",
    reference_y_column="measured_value",
    candidate_x_column="predicted_time",
    candidate_y_column="predicted_value",
    order_column="sample_id",
)
```

`comparison` is indexed by `(reference, candidate, metric)`. `coverage`
records, for every `(reference, candidate)` pair, how many usable points each
side had, whether it was matched or skipped, and why.

If a table interleaves genuine reference measurements with rows that exist
only to carry a candidate value at an extra grid point, pass
`reference_mask_column` naming a boolean column that flags real reference
rows.

## Scoring many candidates across many references

Given a `comparison` table with `(reference, candidate, metric)` index levels
(from `compare_tabular_data` or `compare_curve_set`), you can compute a
robust, standardized score per candidate, following equations 17-18 of the
paper:

```python
from curve_matching import candidate_scores, summarize_comparisons

summary, populations = summarize_comparisons(comparison)
scores, epsilon, sigma = candidate_scores(
    summary, results,
    reference_column="data_set",
    y_column="measured_value",
    sigma_column="measurement_uncertainty",
)
```

- `summarize_comparisons` standardizes each metric across the candidate
  population for every reference (equation 17), averages the four metrics,
  and combines the original, shifted, and aligned stages into one `epsilon`
  score per `(reference, candidate)` pair. If any candidate's optimal shift
  hits the search boundary (`shift_fraction >= 0.5`) for a given reference,
  that reference falls back to using only the unshifted score, since the
  shift search did not converge to an interior optimum.
- `candidate_scores` pivots that into a reference-by-candidate matrix,
  weights each reference by the inverse of its relative measurement
  uncertainty (via `reference_relative_uncertainty`), and returns one
  integrated score per candidate, sorted so the best (lowest) score comes
  first.

Lower-level building blocks (`fit_curve`, `curve_dissimilarities`,
`optimal_shift`, `dimensionless_index`, `aggregate_four_metrics`,
`boxplot_statistics`) are also public if you want to assemble a different
workflow.

## Plotting

`curve_matching.plotting` (needs the `plot` extra) mirrors the tabular API,
with column names and axis labels as explicit parameters:

```python
from curve_matching import plotting

plotting.plot_curve_fits(
    results,
    reference_column="data_set", candidate_column="algorithm",
    x_column="measured_time",
    reference_y_column="measured_value", candidate_y_column="predicted_value",
    xlabel="time [s]", ylabel="signal",
)

plotting.plot_shift_alignment(
    results,
    reference_column="data_set", candidate_column="algorithm",
    x_column="measured_time",
    reference_y_column="measured_value", candidate_y_column="predicted_value",
    metric="d_L2_0", xlabel="time [s]", ylabel="signal",
)

plotting.plot_reference_boxplots(comparison)
plotting.plot_candidate_boxplots(comparison)
plotting.plot_candidate_scores(scores)
plotting.plot_metric_bars(comparison, stage="original")
```

## Development

```bash
pip install -e ".[plot]"
pip install pytest pytest-cov
pytest
```

## License

Released under the MIT license; see [LICENSE](LICENSE).

## Citation

The dissimilarity metrics, spline-fitting workflow, and equations 4-18
implemented here come from:

[1] M.S. Bernardi, M. Pelucchi, A. Stagni, L.M. Sangalli, A. Cuoci,
A. Frassoldati, P. Secchi, T. Faravelli, "Curve matching, a generalized
framework for models/experiments comparison: An application to n-heptane
combustion kinetic mechanisms," *Combustion and Flame*, vol. 168, pp.
186-203, 2016. https://doi.org/10.1016/j.combustflame.2016.03.019

```bibtex
@article{BERNARDI2016186,
  title = {Curve matching, a generalized framework for models/experiments comparison: An application to n-heptane combustion kinetic mechanisms},
  journal = {Combustion and Flame},
  volume = {168},
  pages = {186-203},
  year = {2016},
  issn = {0010-2180},
  doi = {https://doi.org/10.1016/j.combustflame.2016.03.019},
  author = {M.S. Bernardi and M. Pelucchi and A. Stagni and L.M. Sangalli and A. Cuoci and A. Frassoldati and P. Secchi and T. Faravelli},
}
```
