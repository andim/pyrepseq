from pyrepseq import tcr_metric
import numpy as np
from pandas import DataFrame
import pytest


@pytest.mark.parametrize(
    ("metric", "expected"),
    (
        (tcr_metric.BetaCdr3Tcrdist(), np.array([[0, 23, 0], [23, 0, 23]])),
        (tcr_metric.BetaTcrdist(), np.array([[0, 69, 46], [69, 0, 115]])),
    )
)
def test_calc_cdist_matrix(metric, expected, mock_data_df: DataFrame):
    anchor_tcrs = mock_data_df.iloc[0:2]
    comparison_tcrs = mock_data_df.iloc[0:3]

    result = metric.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)

    assert np.array_equal(result, expected)
