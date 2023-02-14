import pandas as pd
from pyrepseq.io import *
import pytest


@pytest.mark.filterwarnings('ignore: Failed to standardise')
def test_standardise():
    df_old = pd.DataFrame(
        [
            ['TRAJ23*01', 'CATQYF', 'TRAV26-1*01', 'TRBJ2-3*01', 'CASQYF', 'TRBV13*01', 'AAA', 'A1', 'B2M', 1],
            [None, 'CATQYF', None, None, 'CASQYF', None, None, None, None, 1],
            ['foobar', 'CATQYF', 'TRAV26-1*01', 'TRBJ2-3*01', 'CASQYF', 'TRBV13*01', 'AAA', 'foobar', 'B2M', 1]
        ],
        columns=range(10)
    )
    expected = pd.DataFrame(
        [
            ['TRAJ23', 'CATQYF', 'TRAV26-1', 'TRBJ2-3', 'CASQYF', 'TRBV13', 'AAA', 'HLA-A', 'B2M', 1],
            [None, 'CATQYF', None, None, 'CASQYF', None, None, None, None, 1],
            [None, 'CATQYF', 'TRAV26-1', 'TRBJ2-3', 'CASQYF', 'TRBV13', 'AAA', None, 'B2M', 1]
        ],
        columns=["TRAV", "CDR3A","TRAJ", "TRBV", "CDR3B", "TRBJ", "Epitope", "MHCA", "MHCB", "clonal_counts"]
    )

    result = standardize_dataframe(
        df_old=df_old,
        from_columns=range(10)
    )

    assert result.equals(expected)