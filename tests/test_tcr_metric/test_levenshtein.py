from pyrepseq import tcr_metric
import numpy as np
from pandas import DataFrame
import pytest


@pytest.mark.parametrize(
    ("metric", "expected"),
    (
        (tcr_metric.AlphaCdr3Levenshtein(), np.array([[0, 4, 0], [4, 0, 4]])),
        (tcr_metric.BetaCdr3Levenshtein(), np.array([[0, 6, 0], [6, 0, 6]])),
        (tcr_metric.Cdr3Levenshtein(), np.array([[0, 10, 0], [10, 0, 10]])),
        (tcr_metric.AlphaCdrLevenshtein(), np.array([[0, 4, 12], [4, 0, 16]])),
        (tcr_metric.BetaCdrLevenshtein(), np.array([[0, 6, 7], [6, 0, 13]])),
        (tcr_metric.CdrLevenshtein(), np.array([[0, 10, 19], [10, 0, 29]]))
    )
)
def test_calc_cdist_matrix(metric, expected, mock_data_df: DataFrame):
    anchor_tcrs = mock_data_df.iloc[0:2]
    comparison_tcrs = mock_data_df.iloc[0:3]

    result = metric.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)

    assert np.array_equal(result, expected)
