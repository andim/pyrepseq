import pandas as pd
from pathlib import Path
import pytest


RESOURCES_DIR = Path("tests")/"resources"


@pytest.fixture
def mock_data_df():
    return pd.read_csv(RESOURCES_DIR/"mock_data.csv")