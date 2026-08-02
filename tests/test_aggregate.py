import numpy as np
import pandas as pd
import pytest

from curve_matching import (
    METRIC_NAMES,
    candidate_scores,
    dimensionless_index,
    summarize_comparisons,
)
from curve_matching.api import compare_tabular_data


def test_equation_17_rejects_a_single_candidate_population():
    with pytest.raises(ValueError, match="at least two candidates"):
        dimensionless_index(pd.DataFrame([[1.0] * 4], columns=METRIC_NAMES))


def test_maximum_shift_reference_uses_original_only_score():
    index = pd.MultiIndex.from_product(
        [["r1"], ["A", "B"], METRIC_NAMES],
        names=["reference", "candidate", "metric"],
    )
    table = pd.DataFrame(index=index)
    table["original"] = np.tile([0.1, 0.2, 0.3, 0.4], 2) * np.repeat([1.0, 1.3], 4)
    table["shift_fraction"] = np.tile([0.1, 0.2, 0.3, 0.5], 2)
    table["shift_percent"] = 100.0 * table["shift_fraction"]
    table["delta"] = table["shift_fraction"]
    table["aligned"] = table["original"] / 2.0
    summary, _ = summarize_comparisons(table)
    assert summary["used_original_only"].all()
    assert summary["epsilon_all"].isna().all()
    assert np.allclose(summary["epsilon"], summary["epsilon_original"])


def test_candidate_scores_ranks_the_closer_candidate_first():
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
                    "sigma": 1.0,
                    "predicted": value + offset * (1.0 + 3.0 * (position - 1.0)),
                }
            )
    data = pd.DataFrame(rows)
    comparisons, _ = compare_tabular_data(
        data,
        reference_column="reference",
        candidate_column="candidate",
        reference_x_column="position",
        reference_y_column="measured",
        candidate_x_column="position",
        candidate_y_column="predicted",
        order_column="point",
    )
    summary, _ = summarize_comparisons(comparisons)
    scores, epsilon, sigma = candidate_scores(
        summary, data,
        reference_column="reference", y_column="measured", sigma_column="sigma",
    )
    assert set(scores["candidate"]) == {"A", "B"}
    assert scores.set_index("candidate").loc["A", "score"] < scores.set_index("candidate").loc["B", "score"]
    assert epsilon.shape == (1, 2)
    assert (sigma > 0).all()
