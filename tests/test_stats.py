import pandas as pd
import numpy as np
import pyrepseq as prs
from pytest import mark


@mark.filterwarnings("ignore:Inputting paired-chain CDR3 data as a tuple")
@mark.parametrize(
    ("arg", "expected"),
    (
        (["A", "A"], 1.0),
        (["A", "B"], 0.0),
        (["A", "A", "B"], 1.0 / 3.0),
        ((["A","A"],["B","B"]), 1.0),
        ((["A","B"],["A","B"]), 0.0),
        ((["A","A","B"],["C","C","C"]), 1.0 / 3.0)
    )
)
def test_with_one_arg(arg, expected):
    result = prs.pc(arg)
    assert result == expected


def test_with_one_df(mock_data_df):
    # Do we want pc to detect clone count column and automatically compute based on that?
    # Currently it just looks for duplicate rows and calculates based on that.
    mock_data_df_with_coincidence = pd.concat([mock_data_df,mock_data_df.iloc[[-1]]])
    result = prs.pc(mock_data_df_with_coincidence)
    num_items = len(mock_data_df_with_coincidence)
    num_pairs = num_items * (num_items-1) / 2
    expected = 1.0 / num_pairs
    assert result == expected


@mark.filterwarnings("ignore:Inputting paired-chain CDR3 data as a tuple")
@mark.parametrize(
    ("arg1", "arg2", "expected"),
    (
        (["A", "A"], ["A", "A"], 1.0),
        (["A", "A"], ["B", "B"], 0.0),
        (["A", "B"], ["A", "B"], 0.5),
        ((["A","A"],["B","B"]), (["A","A"],["B","B"]), 1.0),
        ((["A","B"],["A","B"]), (["A","B"],["A","B"]), 0.5)
    )
)
def test_with_two_args(arg1, arg2, expected):
    result = prs.pc(arg1, arg2)
    assert result == expected


def test_with_two_dfs(mock_data_df):
    result = prs.pc(mock_data_df, mock_data_df)
    expected = 1 / len(mock_data_df)
    assert result == expected

df_numeric = pd.DataFrame(columns=['A', 'B'], data=[[1, 1],
                                                [1, 1],
                                                [1, 2]])
df_string = pd.DataFrame(columns=['A', 'B'], data=[['1', '1'],
                                            ['1', '1'],
                                            ['1', '2']])
@mark.parametrize("df", [df_numeric, df_string])
def test_pc_joint(df):
    assert prs.pc_joint(df, on=['A', 'B']) == 1/3.

@mark.parametrize("df", [df_numeric, df_string])
def test_pc_conditional(df):
    assert prs.pc_conditional(df, by='A', on='B') == 1/3.
    assert prs.pc_conditional(df, by='B', on='A') == 1.0

@mark.parametrize("df", [df_numeric, df_string])
def test_renyi2(df):
    assert prs.renyi2_entropy(df, features=['A', 'B']) == -np.log2(1/3.)
    assert prs.renyi2_entropy(df, features='A', by='B') == 0
