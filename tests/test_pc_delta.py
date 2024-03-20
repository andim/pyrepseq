import numpy as np
import pyrepseq as prs
from pytest import mark


@mark.filterwarnings("ignore:Inputting paired-chain CDR3 data as a tuple")
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


def test_with_one_arg_with_downsampling():
    results = prs.pcDelta(["A", "A", "A"], bins=range(5), maxseqs=2)
    assert np.array_equal(results, np.array([1,0,0,0]))


def test_with_one_df(mock_data_df):
    results = prs.pcDelta(mock_data_df, bins=range(12))
    expected = np.array([1.0/3.0,0,0,0,0,0,0,0,0,0,2.0/3.0])
    assert np.array_equal(results, expected)


@mark.filterwarnings("ignore:Inputting paired-chain CDR3 data as a tuple")
@mark.parametrize(
    ("arg1", "arg2", "expected"),
    (
        (["A","A"], ["A","A"], np.array([1,0,0,0])),
        (["A","A"], ["B","B"], np.array([0,1,0,0])),
        (["A","B"], ["A","B"], np.array([0.5,0.5,0,0])),
        ((["A","A"],["B","B"]), (["A","A"],["B","B"]), np.array([1,0,0,0])),
        ((["A","B"],["A","B"]), (["A","B"],["A","B"]), np.array([0.5,0,0.5,0]))
    )
)
def test_with_two_args(arg1, arg2, expected):
    results = prs.pcDelta(arg1, arg2, bins=range(5))
    assert np.array_equal(results, expected)


def test_with_two_dfs(mock_data_df):
    results = prs.pcDelta(mock_data_df, mock_data_df, bins=range(12))
    expected = np.array([5.0/9.0,0,0,0,0,0,0,0,0,0,4.0/9.0])
    assert np.array_equal(results, expected)