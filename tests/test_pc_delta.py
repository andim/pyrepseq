import numpy as np
import pyrepseq as prs
from pytest import mark


@mark.parametrize(
    ("arg", "expected"),
    (
        (["A", "A"], np.array([1,0,0,0])),
        (["A", "B"], np.array([0,1,0,0])),
        (["A", "A", "B"], np.array([1.0/3.0,2.0/3.0,0,0])),
        ((["A","A"],["B","B"]), np.array([1,0,0,0])),
        ((["A", "B"], ["A", "B"]), np.array([0,0,1,0]))
    )
)
def test_with_one_arg(arg, expected):
    results = prs.pcDelta(arg, bins=range(5))
    assert np.array_equal(results, expected)


def test_with_one_df(mock_data_df):
    results = prs.pcDelta(mock_data_df, bins=range(12))
    expected = np.array([1.0/3.0,0,0,0,0,0,0,0,0,0,2.0/3.0])
    assert np.array_equal(results, expected)