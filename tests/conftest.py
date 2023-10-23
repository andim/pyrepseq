import pandas as pd
from pathlib import Path
import pytest


@pytest.fixture
def mock_data_path():
    return Path("tests") / "resources" / "mock_data.csv"


@pytest.fixture
def mock_data_df(mock_data_path):
    return pd.read_csv(mock_data_path)
