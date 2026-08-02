import numpy as np
import pandas as pd
import pytest

from curve_matching import METRIC_NAMES, compare_curve_data, compare_curve_set, compare_tabular_data


def test_compare_curve_data_supports_different_grids():
    x_reference = np.linspace(0.0, 1.0, 12)
    y_reference = np.sin(2.0 * np.pi * x_reference)
    x_candidate = np.linspace(0.0, 1.0, 20)
    y_candidate = np.sin(2.0 * np.pi * x_candidate) + 0.02

    comparison = compare_curve_data(
        x_reference, y_reference, x_candidate, y_candidate,
        reference_name="measurement", candidate_name="simulation",
    )
    assert set(comparison.index) == set(METRIC_NAMES)
    assert (comparison["original"] > 0.0).all()


def test_compare_curve_set_returns_one_row_per_candidate_per_metric():
    x = np.linspace(0.0, 1.0, 15)
    y_reference = np.sin(2.0 * np.pi * x)
    # A fixed-shape distortion (different curvature than the reference) at
    # two magnitudes, so every metric -- amplitude and derivative alike --
    # grows monotonically with the perturbation size.
    distortion = np.cos(6.0 * np.pi * x)
    comparison = compare_curve_set(
        x, y_reference,
        {
            "model-a": (x, y_reference + 0.01 * distortion),
            "model-b": (x, y_reference + 0.10 * distortion),
        },
    )
    assert comparison.index.names == ["candidate", "metric"]
    assert set(comparison.index.get_level_values("candidate")) == {"model-a", "model-b"}
    a = comparison.loc["model-a"]["original"]
    b = comparison.loc["model-b"]["original"]
    assert (b > a).all()


def test_compare_curve_set_rejects_empty_candidates():
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="at least one"):
        compare_curve_set(x, np.sin(x), {})


def test_compare_tabular_data_matches_three_point_series():
    rows = []
    for candidate, offset in (("A", 0.0), ("B", 1.0)):
        for order, (position, value) in enumerate(((0.8, 20.0), (1.0, 35.0), (1.2, 30.0))):
            rows.append(
                {
                    "reference": "series-1",
                    "candidate": candidate,
                    "point": order,
                    "position": position,
                    "measured": value,
                    "predicted": value + offset * (1.0 + 3.0 * (position - 1.0)),
                }
            )
    data = pd.DataFrame(rows)
    comparisons, coverage = compare_tabular_data(
        data,
        reference_column="reference",
        candidate_column="candidate",
        reference_x_column="position",
        reference_y_column="measured",
        candidate_x_column="position",
        candidate_y_column="predicted",
        order_column="point",
    )
    assert len(comparisons) == 2 * len(METRIC_NAMES)
    assert (coverage["status"] == "matched").all()


def test_compare_tabular_data_reports_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        compare_tabular_data(
            pd.DataFrame({"reference": ["r"], "candidate": ["c"]}),
            reference_column="reference",
            candidate_column="candidate",
            reference_x_column="x",
            reference_y_column="y_ref",
            candidate_x_column="x",
            candidate_y_column="y_cand",
        )


def test_compare_tabular_data_skips_sparse_groups():
    data = pd.DataFrame(
        {
            "reference": ["r1", "r1"],
            "candidate": ["c1", "c1"],
            "x": [0.0, 1.0],
            "y_ref": [0.0, 1.0],
            "y_cand": [0.0, 1.0],
        }
    )
    comparisons, coverage = compare_tabular_data(
        data,
        reference_column="reference",
        candidate_column="candidate",
        reference_x_column="x",
        reference_y_column="y_ref",
        candidate_x_column="x",
        candidate_y_column="y_cand",
        min_points=3,
    )
    assert comparisons.empty
    assert (coverage["status"] == "skipped").all()
    assert "fewer than 3" in coverage["reason"].iloc[0]
